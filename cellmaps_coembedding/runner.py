#! /usr/bin/env python

import os
import csv
import random
import logging
import time
from cellmaps_utils import logutils
import cellmaps_coembedding
from cellmaps_coembedding import muse_sc as muse
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError

logger = logging.getLogger(__name__)


class CellmapsCoEmbeddingRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None, image_embedding=None,
                 apms_embedding=None,
                 image_dimension=1024,
                 apms_dimension=1024,
                 latent_dimension=128,
                 skip_logging=False,
                 misc_info_dict=None):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsCoEmbeddingRunner.run` method
        :type int:
        """
        self._outdir = outdir
        self._start_time = int(time.time())
        self._end_time = -1
        self._image_embedding = image_embedding
        self._apms_embedding = apms_embedding
        self._latent_dimension = latent_dimension
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

            imgdim = 1024  # input dim of image
            ppidim = 1024  # input dim of ppi
            latent_dim = 128  # output dim of music embedding
            k = 10  # k nearest neighbors value used for clustering - clustering used for triplet loss
            min_diff = 0.2  # margin for triplet loss
            dropout = 0.25  # dropout between neural net layers
            n_epochs = 500  # training epochs
            lambda_regul = 5  # weight for regularization term in loss
            lambda_super = 5  # weight for triplet loss term in loss

            # TODO: Fix this
            """
            muse.muse_fit_predict(resultsdir=os.path.join(self._outdir, 'music_emd.tsv'),
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
