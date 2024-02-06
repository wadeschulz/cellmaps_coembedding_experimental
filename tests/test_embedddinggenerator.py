#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `EmbeddingGenerator` class."""

import os
import shutil
import unittest
import tempfile
from unittest.mock import MagicMock

from cellmaps_utils.exceptions import CellMapsProvenanceError
from cellmaps_utils import constants

from cellmaps_coembedding.runner import EmbeddingGenerator
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError


class TestEmbeddingGenerator(unittest.TestCase):
    """Tests for `EmbeddingGenerator` class."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_get_embedding_file_no_embedding_file_found(self):
        e_gen = EmbeddingGenerator()
        temp_dir = tempfile.mkdtemp()
        try:
            e_gen._get_embedding_file(temp_dir)
            self.fail('expected exception')
        except CellmapsCoEmbeddingError as ce:
            self.assertTrue('Embedding file not found in ' in str(ce))
        finally:
            shutil.rmtree(temp_dir)

    def test_get_embedding_file_ppi_file_found(self):
        e_gen = EmbeddingGenerator()
        temp_dir = tempfile.mkdtemp()
        try:
            ppi_file = os.path.join(temp_dir,
                                    constants.PPI_EMBEDDING_FILE)
            open(ppi_file, 'a').close()
            self.assertEqual(e_gen._get_embedding_file(temp_dir),
                             ppi_file)

        finally:
            shutil.rmtree(temp_dir)

    def test_get_embedding_file_image_file_found(self):
        e_gen = EmbeddingGenerator()
        temp_dir = tempfile.mkdtemp()
        try:
            image_file = os.path.join(temp_dir,
                                      constants.IMAGE_EMBEDDING_FILE)
            open(image_file, 'a').close()
            self.assertEqual(e_gen._get_embedding_file(temp_dir),
                             image_file)

        finally:
            shutil.rmtree(temp_dir)
