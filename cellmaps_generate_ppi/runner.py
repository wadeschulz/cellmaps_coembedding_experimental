#! /usr/bin/env python

import os
import csv
import random
import logging
from cellmaps_generate_ppi.exceptions import CellmapsgenerateppiError

logger = logging.getLogger(__name__)


class CellmapsgenerateppiRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None, image_embedding=None,
                 apms_embedding=None,
                 latent_dimension=None):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsgenerateppiRunner.run` method
        :type int:
        """
        self._outdir = outdir
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
            raise CellmapsgenerateppiError('outdir must be set')

        if not os.path.isdir(self._outdir):
            os.makedirs(self._outdir, mode=0o755)

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
        return 0
