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


class EmbeddingGenerator(object):
    """
    Base class for implementations that generate
    network embeddings
    """
    def __init__(self, dimensions=1024,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 ):
        """
        Constructor
        """
        self._dimensions = dimensions
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
                 k=10, triplet_margin=0.1, dropout=0.25, n_epochs=500,
                 n_epochs_init=200,
                 outdir=None,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 jackknife_percent = 0
                ):
        """

        :param dimensions:
        :param k: k nearest neighbors value used for clustering - clustering used for triplet loss
        :param triplet_margin: margin for triplet loss
        :param dropout: dropout between neural net layers
        :param n_epochs: training epochs
        :param n_epochs_init: initialization training epochs
        :param outdir:
        :param ppi_embeddingdir:
        :param image_embeddingdir:
        :param jackknife_percent: percent of data to withhold from training
        """
        super().__init__(dimensions=dimensions,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir)
        self._outdir = outdir
        self._k = k
        self.triplet_margin = triplet_margin
        self._dropout = dropout
        self._n_epochs = n_epochs
        self._n_epochs_init = n_epochs_init
        self._jackknife_percent = jackknife_percent

    def get_next_embedding(self):
        """

        :return:
        """
        ppi_embeddings = self._get_ppi_embeddings()
        ppi_embeddings.sort(key=lambda x: x[0])
        logger.info('There are ' + str(len(ppi_embeddings)) + ' PPI embeddings')
        image_embeddings = self._get_image_embeddings()
        logger.info('There are ' + str(len(image_embeddings)) + ' image embeddings')
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

        test_subset = random.sample(list(np.arange(len(name_index))), int(self._jackknife_percent * len(name_index)))
        if self._jackknife_percent > 0:
            with open('{}_test_genes.txt'.format(resultsdir), 'w') as file:
                file.write('\n'.join(np.array(name_index)[test_subset]))

        model, res_embedings = muse.muse_fit_predict(resultsdir=resultsdir,
                                                     data_x=ppi_embeddings_array,
                                                     data_y=image_embeddings_array,
                                                     name_index=name_index,
                                                     test_subset = test_subset,
                                                     latent_dim=self.get_dimensions(),
                                                     n_epochs=self._n_epochs,
                                                     n_epochs_init=self._n_epochs_init,
                                                     triplet_margin=self.triplet_margin,
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
                 image_embeddingdir=None):
        """
        Constructor
        :param dimensions:
        """
        super().__init__(dimensions=dimensions,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir)

    def get_next_embedding(self):
        """
        Gets next embedding

        :return:
        """
        ppi_embeddings = self._get_ppi_embeddings()
        logger.info('There are ' + str(len(ppi_embeddings)) + ' PPI embeddings')
        image_embeddings = self._get_image_embeddings()
        logger.info('There are ' + str(len(image_embeddings)) + ' image embeddings')

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
                 skip_logging=True,
                 input_data_dict=None):
        """
        Constructor
        :param outdir: Directory to write the results of this tool
        :type outdir: str
        :param inputdir: Output directory where embeddings to be coembedded are located
                         (output of cellmaps_image_embedding and cellmaps_ppi_embedding)
        :type inputdir: str
        :param embedding_generator:
        :param skip_logging: If ``True`` skip logging, if ``None`` or ``False`` do NOT skip logging
        :type skip_logging: bool
        :param name:
        :type name: str
        :param organization_name:
        :type organization_name: str
        :param project_name:
        :type project_name: str
        :param input_data_dict:
        :type input_data_dict: dict
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
        self._keywords = None
        self._description = None
        self._embedding_generator = embedding_generator
        self._input_data_dict = input_data_dict
        self._softwareid = None
        self._coembedding_id = None

        if skip_logging is None:
            self._skip_logging = False
        else:
            self._skip_logging = skip_logging

        logger.debug('In constructor')

    def _update_provenance_fields(self):
        """

        :return:
        """
        prov_attrs = self._provenance_utils.get_merged_rocrate_provenance_attrs(self._inputdirs,
                                                                                override_name=self._name,
                                                                                override_project_name=self._project_name,
                                                                                override_organization_name=self._organization_name,
                                                                                extra_keywords=['merged embedding'])

        self._name = prov_attrs.get_name()
        self._organization_name = prov_attrs.get_organization_name()
        self._project_name = prov_attrs.get_project_name()
        self._keywords = prov_attrs.get_keywords()
        self._description = prov_attrs.get_description()

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
        try:
            self._provenance_utils.register_rocrate(self._outdir,
                                                    name=self._name,
                                                    organization_name=self._organization_name,
                                                    project_name=self._project_name,
                                                    description=self._description,
                                                    keywords=self._keywords)
        except TypeError as te:
            raise CellmapsCoEmbeddingError('Invalid provenance: ' + str(te))
        except KeyError as ke:
            raise CellmapsCoEmbeddingError('Key missing in provenance: ' + str(ke))

    def _register_software(self):
        """
        Registers this tool

        :raises CellMapsImageEmbeddingError: If fairscape call fails
        """
        software_keywords = self._keywords
        software_keywords.extend(['tools', cellmaps_coembedding.__name__])
        software_description = self._description + ' ' + \
                               cellmaps_coembedding.__description__
        self._softwareid = self._provenance_utils.register_software(self._outdir,
                                                                    name=cellmaps_coembedding.__name__,
                                                                    description=software_description,
                                                                    author=cellmaps_coembedding.__author__,
                                                                    version=cellmaps_coembedding.__version__,
                                                                    file_format='py',
                                                                    keywords=software_keywords,
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

        keywords = self._keywords
        keywords.extend(['computation'])
        description = self._description + ' run of ' + cellmaps_coembedding.__name__

        self._provenance_utils.register_computation(self._outdir,
                                                    name=cellmaps_coembedding.__computation_name__,
                                                    run_by=str(self._provenance_utils.get_login()),
                                                    command=str(self._input_data_dict),
                                                    description=description,
                                                    keywords=keywords,
                                                    used_software=[self._softwareid],
                                                    used_dataset=used_dataset,
                                                    generated=[self._coembedding_id])

    def _register_image_coembedding_file(self):
        """
        Registers coembedding file with create as a dataset

        """
        description = self._description
        description += ' Co-Embedding file'
        keywords = self._keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(self.get_coembedding_file()) + ' coembedding output file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'tsv',
                     'author': cellmaps_coembedding.__name__,
                     'version': cellmaps_coembedding.__version__,
                     'date-published': date.today().strftime(self._provenance_utils.get_default_date_format_str())}
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

            self._update_provenance_fields()
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
            # write a task finish file
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            end_time=self._end_time,
                                            status=exitcode)
        logger.debug('Exit code: ' + str(exitcode))
        return exitcode
