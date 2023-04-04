#! /usr/bin/env python

import os
import csv
import random
import logging
import time
from cellmaps_utils import cellmaps_io
import cellmaps_generate_ppi
from cellmaps_generate_ppi.exceptions import CellmapsGenerateppiError

logger = logging.getLogger(__name__)


class CellmapsGenerateppiRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None, image_embedding=None,
                 apms_embedding=None,
                 latent_dimension=None):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsGenerateppiRunner.run` method
        :type int:
        """
        self._outdir = outdir
        self._start_time = int(time.time())
        self._image_embedding = image_embedding
        self._apms_embedding = apms_embedding
        self._latent_dimension = latent_dimension
        logger.debug('In constructor')

    def run(self):
        """
        Runs CM4AI Generate PPI


        :return:
        """
        logger.debug('In run method')
        if self._outdir is None:
            raise CellmapsGenerateppiError('outdir must be set')

        if not os.path.isdir(self._outdir):
            os.makedirs(self._outdir, mode=0o755)

        cellmaps_io.setup_filelogger(outdir=self._outdir,
                                     handlerprefix='cellmaps_generate_ppi')
        cellmaps_io.write_task_start_json(outdir=self._outdir,
                                          start_time=self._start_time,
                                          data={'image_embedding': str(self._image_embedding),
                                                'apms_embedding': str(self._apms_embedding)},
                                          version=cellmaps_generate_ppi.__version__)

        exit_status = 99
        try:
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
                    for geneb in uniq_genes:
                        if genea == geneb:
                            continue
                        f.write(str(genea) + '\t' + str(geneb) + '\t' +
                                str(random.random()) + '\n')
            exit_status = 0
        finally:
            cellmaps_io.write_task_finish_json(outdir=self._outdir,
                                               start_time=self._start_time,
                                               status=exit_status)
        return exit_status
