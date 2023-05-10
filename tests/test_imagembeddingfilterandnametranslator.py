#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_coembedding` package."""

import os
import csv
import shutil
import tempfile
import unittest
from cellmaps_utils import constants
from cellmaps_coembedding.runner import ImageEmbeddingFilterAndNameTranslator
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError


class TestImageEmbeddingFilterAndNameTranslator(unittest.TestCase):
    """Tests for `ImageEmbeddingFilterAndNameTranslator` class."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_with_valid_attrs_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            attr_file = os.path.join(temp_dir, constants.IMAGE_GENE_NODE_ATTR_FILE)
            with open(attr_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, delimiter='\t',
                                        fieldnames=constants.IMAGE_GENE_NODE_COLS)
                writer.writeheader()
                writer.writerow({'name': 'STAG2',
                                 'ambiguous': '',
                                 'represents': 'ensembl:ENSG00000101972',
                                 'antibody': 'HPA002857',
                                 'filename': '2017_E1_2_,2017_E1_1_'})
                writer.writerow({'name': 'RPL22L1',
                                 'ambiguous': 'RPL22,RPL22L1',
                                 'represents': 'ensembl:ENSG00000163584',
                                 'antibody': 'HPA068294,HPA048060',
                                 'filename': '1542_A12_2_,1542_A12_1_,'
                                             '1270_C11_2_,1270_C11_1_'})

            translator = ImageEmbeddingFilterAndNameTranslator(image_downloaddir=temp_dir)
            self.assertEqual({'2017_E1_2_': 'STAG2',
                              '1542_A12_2_': 'RPL22L1'},
                             translator.get_name_mapping())

            # try with embedding
            embedding = [['1_2_3', 0, 1],
                         ['2017_E1_2_', 0, 1],
                         ['1542_A12_2_', 0, 1]]
            res = translator.translate(embedding)
            self.assertEqual([['1_2_3', 0, 1],
                              ['2017_E1_2_', 0, 1],
                              ['1542_A12_2_', 0, 1]], embedding)

        finally:
            shutil.rmtree(temp_dir)
