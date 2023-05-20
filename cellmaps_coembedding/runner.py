#! /usr/bin/env python

import os
import csv
import random
import logging
import time
from datetime import date
import pandas as pd
import numpy as np
import dill
import sys
from tqdm import tqdm
from cellmaps_utils import constants
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil
import cellmaps_coembedding
import cellmaps_coembedding.muse_sc as muse
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError

logger = logging.getLogger(__name__)


class ImageEmbeddingFilterAndNameTranslator(object):
    """
    Converts image embedding names and filters keeping only
    one per gene

    """

    def __init__(self, image_downloaddir=None):
        """
        Constructor
        """
        self._id_to_gene_mapping = self._gen_filtered_mapping(os.path.join(image_downloaddir,
                                                                           constants.IMAGE_GENE_NODE_ATTR_FILE))

    def _gen_filtered_mapping(self, image_gene_node_attrs_file):
        """
        Reads TSV file

        :param image_gene_node_attrs_file:
        :return:
        """
        mapping_dict = {}
        with open(image_gene_node_attrs_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                mapping_dict[row['filename'].split(',')[0]] = row['name']
        return mapping_dict

    def get_name_mapping(self):
        """
        Gets mapping of old name to new name

        :return: mapping of old name to new name
        :rtype: dict
        """
        return self._id_to_gene_mapping

    def translate(self, embeddings):
        """
        Translates and filters out embeddings
        with duplicate gene names. Updated embeddings are returned
        :param embeddings: list of list of embeddings
        :type embeddings: list
        :return: embeddings
        :rtype: list
        """
        res_embeddings = []

        for row in embeddings:
            if row[0] not in self._id_to_gene_mapping:
                continue
            new_row = row.copy()
            new_row[0] = self._id_to_gene_mapping[row[0]]
            res_embeddings.append(new_row)
        return res_embeddings


class EmbeddingGenerator(object):
    """
    Base class for implementations that generate
    network embeddings
    """
    def __init__(self, dimensions=1024,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 image_downloaddir=None,
                 img_emd_translator=None):
        """
        Constructor
        """
        self._dimensions = dimensions
        if img_emd_translator is None:
            self._img_emd_translator = ImageEmbeddingFilterAndNameTranslator(image_downloaddir=image_downloaddir)
        self._ppi_embeddingdir = ppi_embeddingdir
        self._image_embeddingdir = image_embeddingdir

    def _get_ppi_embeddings_file(self):
        """

        :return:
        """
        return os.path.join(self._ppi_embeddingdir,
                            constants.PPI_EMBEDDING_FILE)

    def _get_ppi_embeddings(self):
        """
        Gets PPI embedding from ppi embedding directory set via input

        :return: embeddings
        :rtype: list
        """
        return self._get_embedding(self._get_ppi_embeddings_file())

    def _get_image_embeddings_file(self):
        """

        :return:
        """
        return os.path.join(self._image_embeddingdir,
                            constants.IMAGE_EMBEDDING_FILE)

    def _get_image_embeddings(self):
        """
        Gets PPI embedding from ppi embedding directory set via input

        :return: embeddings
        :rtype: list
        """
        return self._get_embedding(self._get_image_embeddings_file())

    def _get_set_of_embedding_names(self, embedding):
        """
        Get a set of embedding names from **embedding**

        :param embedding:
        :return:
        """
        name_set = set()
        for entry in embedding:
            name_set.add(entry[0])
        return name_set

    def _get_embedding(self, embedding_file):
        """
        Gets embedding as a list or lists

        :param embedding_file: Path to embedding file
        :type embedding_file: str
        :return: embeddings
        :rtype: list
        """
        embeddings = []
        with open(embedding_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                embeddings.append(row)
        return embeddings

    def get_dimensions(self):
        """
        Gets number of dimensions this embedding will generate

        :return: number of dimensions aka vector length
        :rtype: int
        """
        return self._dimensions

    def get_name_mapping(self):
        """

        :return:
        """
        return self._img_emd_translator.get_oldname_to_new_name_mapping()

    def get_next_embedding(self):
        """
        Generator method for getting next embedding.
        Caller should implement with ``yield`` operator

        :raises: NotImplementedError: Subclasses should implement this
        :return: Embedding
        :rtype: list
        """
        raise NotImplementedError('Subclasses should implement')


class MuseCoEmbeddingGenerator(EmbeddingGenerator):
    """
    Generats co-embedding using MUSE
    """
    def __init__(self, dimensions=128,
                 k=10, min_diff=0.2, dropout=0.25, n_epochs=500,
                 n_epochs_init=200,
                 outdir=None,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 image_downloaddir=None,
                 img_emd_translator=None):
        """

        :param dimensions:
        :param k: k nearest neighbors value used for clustering - clustering used for triplet loss
        :param min_diff: margin for triplet loss
        :param dropout: dropout between neural net layers
        :param n_epochs: training epochs
        :param n_epochs_init: initialization training epochs
        :param outdir:
        :param ppi_embeddingdir:
        :param image_embeddingdir:
        :param image_downloaddir:
        :param img_emd_translator:
        """
        super().__init__(dimensions=dimensions,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir,
                         image_downloaddir=image_downloaddir,
                         img_emd_translator=img_emd_translator)
        self._outdir = outdir
        self._k = k
        self._min_diff = min_diff
        self._dropout = dropout
        self._n_epochs = n_epochs
        self._n_epochs_init = n_epochs_init

    def get_next_embedding(self):
        """

        :return:
        """
        ppi_embeddings = self._get_ppi_embeddings()
        ppi_embeddings.sort(key=lambda x: x[0])
        logger.info('There are ' + str(len(ppi_embeddings)) + ' PPI embeddings')
        raw_embeddings = self._get_image_embeddings()
        logger.info('There are ' + str(len(raw_embeddings)) + ' raw image embeddings')

        image_embeddings = self._img_emd_translator.translate(raw_embeddings)
        logger.info('There are ' + str(len(image_embeddings)) +
                    ' translated and filtered image embeddings')
        image_embeddings.sort(key=lambda x: x[0])
        ppi_name_set = self._get_set_of_embedding_names(ppi_embeddings)
        image_name_set = self._get_set_of_embedding_names(image_embeddings)
        intersection_name_set = ppi_name_set.intersection(image_name_set)
        logger.info('There are ' +
                    str(len(intersection_name_set)) +
                    ' overlapping embeddings')

        name_index = [x[0] for x in ppi_embeddings if x[0] in intersection_name_set]

        ppi_embeddings_array = np.array([np.array([float(e) for e in xi[1:]]) for xi in ppi_embeddings if xi[0] in intersection_name_set])
        image_embeddings_array = np.array([np.array([float(e) for e in xi[1:]]) for xi in image_embeddings if xi[0] in intersection_name_set])

        resultsdir = os.path.join(self._outdir, 'muse')

        model, res_embedings = muse.muse_fit_predict(resultsdir=resultsdir,
                                                     index=name_index, data_x=ppi_embeddings_array,
                                                     data_y=image_embeddings_array,
                                                     latent_dim=self.get_dimensions(),
                                                     n_epochs=self._n_epochs,
                                                     n_epochs_init=self._n_epochs_init,
                                                     min_diff=self._min_diff,
                                                     k=self._k, dropout=self._dropout)
        for index, embedding in enumerate(res_embedings):
            row = [name_index[index]]
            row.extend(embedding)
            yield row


class FakeCoEmbeddingGenerator(EmbeddingGenerator):
    """
    Generates a fake coembedding
    """
    def __init__(self, dimensions=128, ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 image_downloaddir=None,
                 img_emd_translator=None):
        """
        Constructor
        :param dimensions:
        """
        super().__init__(dimensions=dimensions,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir,
                         image_downloaddir=image_downloaddir,
                         img_emd_translator=img_emd_translator)

    def get_next_embedding(self):
        """
        Gets next embedding

        :return:
        """
        ppi_embeddings = self._get_ppi_embeddings()
        logger.info('There are ' + str(len(ppi_embeddings)) + ' PPI embeddings')
        raw_embeddings = self._get_image_embeddings()
        logger.info('There are ' + str(len(raw_embeddings)) + ' raw image embeddings')

        image_embeddings = self._img_emd_translator.translate(raw_embeddings)
        logger.info('There are ' + str(len(image_embeddings)) +
                    ' translated and filtered image embeddings')

        ppi_embedding_names = self._get_set_of_embedding_names(ppi_embeddings)
        image_embedding_names = self._get_set_of_embedding_names(image_embeddings)
        intersection_embedding_names = ppi_embedding_names.intersection(image_embedding_names)
        logger.info('There are ' +
                    str(len(intersection_embedding_names)) +
                    ' overlapping embeddings')
        for embed_name in intersection_embedding_names:
            row = [embed_name]
            row.extend([random.random() for x in range(0, self.get_dimensions())])
            yield row


class CellmapsCoEmbedder(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None,
                 inputdirs=None,
                 embedding_generator=None,
                 name=None,
                 organization_name=None,
                 project_name=None,
                 provenance_utils=ProvenanceUtil(),
                 skip_logging=False,
                 input_data_dict=None):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsCoEmbedder.run` method
        :type int:
        """
        if outdir is None:
            raise CellmapsCoEmbeddingError('outdir is None')
        self._outdir = os.path.abspath(outdir)
        self._inputdirs = inputdirs
        self._start_time = int(time.time())
        self._end_time = -1
        self._name = name
        self._project_name = project_name
        self._organization_name = organization_name
        self._provenance_utils = provenance_utils
        self._embedding_generator = embedding_generator
        self._input_data_dict = input_data_dict
        self._softwareid = None
        self._coembedding_id = None

        if skip_logging is None:
            self._skip_logging = False
        else:
            self._skip_logging = skip_logging

        logger.debug('In constructor')

    def _write_task_start_json(self):
        """
        Writes task_start.json file with information about
        what is to be run

        """
        data = {}

        if self._input_data_dict is not None:
            data['commandlineargs'] = self._input_data_dict

        logutils.write_task_start_json(outdir=self._outdir,
                                       start_time=self._start_time,
                                       version=cellmaps_coembedding.__version__,
                                       data=data)

    def _create_rocrate(self):
        """
        Creates rocrate for output directory

        :raises CellMapsProvenanceError: If there is an error
        """
        name_set = set()
        proj_set = set()
        org_set = set()
        for entry in self._inputdirs:
            name, proj_name, org_name = self._provenance_utils.get_name_project_org_of_rocrate(entry)
            name_set.add(name)
            proj_set.add(proj_name)
            org_set.add(org_name)

        name = '|'.join(list(name_set))
        proj_name = '|'.join(list(proj_set))
        org_name = '|'.join(list(org_set))

        if self._name is not None:
            name = self._name

        if self._organization_name is not None:
            org_name = self._organization_name

        if self._project_name is not None:
            proj_name = self._project_name
        try:
            self._provenance_utils.register_rocrate(self._outdir,
                                                    name=name,
                                                    organization_name=org_name,
                                                    project_name=proj_name)
        except TypeError as te:
            raise CellmapsCoEmbeddingError('Invalid provenance: ' + str(te))
        except KeyError as ke:
            raise CellmapsCoEmbeddingError('Key missing in provenance: ' + str(ke))

    def _register_software(self):
        """
        Registers this tool

        :raises CellMapsImageEmbeddingError: If fairscape call fails
        """
        self._softwareid = self._provenance_utils.register_software(self._outdir,
                                                                    name=cellmaps_coembedding.__name__,
                                                                    description=cellmaps_coembedding.__description__,
                                                                    author=cellmaps_coembedding.__author__,
                                                                    version=cellmaps_coembedding.__version__,
                                                                    file_format='.py',
                                                                    url=cellmaps_coembedding.__repo_url__)

    def _register_computation(self):
        """
        # Todo: added inused dataset, software and what is being generated
        :return:
        """
        logger.debug('Getting id of input rocrate')
        used_dataset = []
        for entry in self._inputdirs:
            used_dataset.append(self._provenance_utils.get_id_of_rocrate(entry))
        self._provenance_utils.register_computation(self._outdir,
                                                    name=cellmaps_coembedding.__name__ + ' computation',
                                                    run_by=str(os.getlogin()),
                                                    command=str(self._input_data_dict),
                                                    description='run of ' + cellmaps_coembedding.__name__,
                                                    used_software=[self._softwareid],
                                                    used_dataset=used_dataset,
                                                    generated=[self._coembedding_id])

    def _register_image_coembedding_file(self):
        """
        Registers coembedding file with create as a dataset

        """
        data_dict = {'name': os.path.basename(self.get_coembedding_file()) + ' coembedding output file',
                     'description': 'CoEmbedding file',
                     'data-format': 'tsv',
                     'author': cellmaps_coembedding.__name__,
                     'version': cellmaps_coembedding.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        self._coembedding_id = self._provenance_utils.register_dataset(self._outdir,
                                                                       source_file=self.get_coembedding_file(),
                                                                       data_dict=data_dict,
                                                                       skip_copy=True)

    def get_coembedding_file(self):
        """
        Gets image embedding file
        :return:
        """
        return os.path.join(self._outdir, constants.CO_EMBEDDING_FILE)

    def run(self):
        """
        Runs CM4AI Generate PPI


        :return:
        """
        logger.debug('In run method')
        exitcode = 99
        try:
            if self._outdir is None:
                raise CellmapsCoEmbeddingError('outdir must be set')

            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir, mode=0o755)

            if self._skip_logging is False:
                logutils.setup_filelogger(outdir=self._outdir,
                                          handlerprefix='cellmaps_coembedding')
                self._write_task_start_json()

            self._create_rocrate()
            self._register_software()

            # generate result
            with open(os.path.join(self._outdir, constants.CO_EMBEDDING_FILE), 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                header_line = ['']
                header_line.extend([x for x in range(1, self._embedding_generator.get_dimensions())])
                writer.writerow(header_line)
                for row in tqdm(self._embedding_generator.get_next_embedding(), desc='Saving embedding'):
                    writer.writerow(row)

            self._register_image_coembedding_file()

            self._register_computation()

            exitcode = 0
        finally:
            self._end_time = int(time.time())
            if self._skip_logging is False:
                # write a task finish file
                logutils.write_task_finish_json(outdir=self._outdir,
                                                start_time=self._start_time,
                                                end_time=self._end_time,
                                                status=exitcode)
        logger.debug('Exit code: ' + str(exitcode))
        return exitcode
