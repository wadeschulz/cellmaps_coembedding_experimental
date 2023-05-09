#! /usr/bin/env python

import os
import csv
import random
import logging
import time
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil
import cellmaps_coembedding
from cellmaps_coembedding import muse_sc as muse
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError

logger = logging.getLogger(__name__)


class ImageEmbeddingFilterAndNameTranslator(object):
    """
    Converts image embedding names and filters keeping only
    one per gene

    """

    def __init__(self, image_embeddingdir=None):
        """
        Constructor
        """
        self._id_to_gene_mapping = self._gen_filtered_mapping(os.path.join(image_embeddingdir,
                                                                           'image_gene_node_attributes.tsv'))

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

    def get_oldname_to_new_name_mapping(self):
        """

        :return:
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
            new_row[0] = self._id_to_gene_mapping(row[0])
            res_embeddings.append(new_row)
        return res_embeddings


class EmbeddingGenerator(object):
    """
    Base class for implementations that generate
    network embeddings
    """
    def __init__(self, dimensions=1024):
        """
        Constructor
        """
        self._dimensions = dimensions

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


class FakeCoEmbeddingGenerator(EmbeddingGenerator):
    """
    Generates a fake coembedding
    """
    def __init__(self, dimensions=128, ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 img_emd_translator=None):
        """
        Constructor
        :param dimensions:
        """
        super().__init__(dimensions=dimensions)
        if img_emd_translator is None:
            self._img_emd_translator = ImageEmbeddingFilterAndNameTranslator(image_embeddingdir=image_embeddingdir)
        self._ppi_embeddingdir = ppi_embeddingdir
        self._image_embeddingdir = image_embeddingdir
        self._img_emd_translator = img_emd_translator

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

    def _get_ppi_embeddings(self):
        """
        Gets PPI embedding from ppi embedding directory set via input

        :return: embeddings
        :rtype: list
        """

        return self._get_embedding(os.path.join(self._ppi_embeddingdir, 'ppi_emd.tsv'))

    def _get_image_embeddings(self):
        """
        Gets PPI embedding from ppi embedding directory set via input

        :return: embeddings
        :rtype: list
        """
        return self._get_embedding(os.path.join(self._image_embeddingdir, 'image_emd.tsv'))

    def get_next_embedding(self):
        """
        Gets next embedding

        :return:
        """
        ppi_embeddings = self._get_ppi_embeddings()
        raw_embeddings = self._get_image_embeddings()
        image_embeddings = self._img_emd_translator.translate(raw_embeddings)

        ppi_embedding_names = self._get_set_of_embedding_names(ppi_embeddings)
        image_embedding_names = self._get_set_of_embedding_names(image_embeddings)
        intersection_embedding_names = ppi_embedding_names.intersection(image_embedding_names)
        for embed_name in intersection_embedding_names:
            row = [embed_name]
            row.extend([random.random() for x in range(0, self.get_dimensions())])
            yield row

    def get_image_embedding_oldname_to_new_name_mapping(self):
        """

        :return:
        """
        return self._img_emd_translator.get_oldname_to_new_name_mapping()


class CellmapsCoEmbeddingRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None,
                 embedding_generator=None,
                 name=cellmaps_coembedding.__name__,
                 organization_name=None,
                 project_name=None,
                 provenance_utils=ProvenanceUtil(),
                 skip_logging=False,
                 misc_info_dict=None):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsCoEmbeddingRunner.run` method
        :type int:
        """
        if outdir is None:
            raise CellmapsCoEmbeddingError('outdir is None')
        self._outdir = os.path.abspath(outdir)
        self._start_time = int(time.time())
        self._end_time = -1
        self._name = name
        self._project_name = project_name
        self._organization_name = organization_name
        self._provenance_utils = provenance_utils
        self._embedding_generator = embedding_generator
        self._misc_info_dict = misc_info_dict
        self._softwareid = None

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

        if self._misc_info_dict is not None:
            data.update(self._misc_info_dict)

        logutils.write_task_start_json(outdir=self._outdir,
                                       start_time=self._start_time,
                                       version=cellmaps_coembedding.__version__,
                                       data=data)

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

            """
            imgdim = 1024  # input dim of image
            ppidim = 1024  # input dim of ppi
            latent_dim = 128  # output dim of music embedding
            k = 10  # k nearest neighbors value used for clustering - clustering used for triplet loss
            min_diff = 0.2  # margin for triplet loss
            dropout = 0.25  # dropout between neural net layers
            n_epochs = 500  # training epochs
            lambda_regul = 5  # weight for regularization term in loss
            lambda_super = 5  # weight for triplet loss term in loss

            os.makedirs(os.path.join(self._outdir, 'muse'), mode=0o755)
            # TODO: Fix this


            muse.muse_fit_predict(resultsdir=os.path.join(self._outdir, 'muse', 'result'),
                                  index=overlapping_proteins, data_x=ppi_features.values,
                                  data_y=image_features.values,
                                  latent_dim=latent_dim,
                                  n_epochs=n_epochs,
                                  min_diff=min_diff,
                                  k=k, dropout=dropout)

            
            uniq_genes = set()
            with open(self._image_embedding, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    uniq_genes.add(row[0])

            with open(self._apms_embedding, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    uniq_genes.add(row[0])
            with open(os.path.join(self._outdir, 'music_edgelist.tsv'), 'w') as f:

                f.write('\t'.join(['GeneA', 'GeneB', 'Weight']) + '\n')
                for genea in uniq_genes:
                    if len(genea) == 0:
                        continue
                    for geneb in uniq_genes:
                        if len(geneb) == 0:
                            continue
                        if genea == geneb:
                            continue
                        f.write(str(genea) + '\t' + str(geneb) + '\t' +
                                str(random.random()) + '\n')
            """
            exitcode = 0
        finally:
            self._end_time = int(time.time())
            if self._skip_logging is False:
                # write a task finish file
                logutils.write_task_finish_json(outdir=self._outdir,
                                                start_time=self._start_time,
                                                end_time=self._end_time,
                                                status=exitcode)
        return exitcode
