#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `EmbeddingGenerator` class."""
import csv
import os
import shutil
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
from cellmaps_utils import constants

from cellmaps_coembedding.runner import EmbeddingGenerator, MuseCoEmbeddingGenerator
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError


class TestEmbeddingGenerator(unittest.TestCase):
    """Tests for `EmbeddingGenerator` class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.temp_dir = tempfile.mkdtemp()
        self.e_gen = EmbeddingGenerator(ppi_embeddingdir=self.temp_dir, image_embeddingdir=self.temp_dir)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.temp_dir)

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

    def test_get_ppi_embeddings_file_default(self):
        ppi_file_path = os.path.join(self.temp_dir, constants.PPI_EMBEDDING_FILE)
        with open(ppi_file_path, 'w') as f:
            f.write('')
        self.assertEqual(self.e_gen._get_ppi_embeddings_file(), ppi_file_path)

    def test_get_image_embeddings_file_default(self):
        image_file_path = os.path.join(self.temp_dir, constants.IMAGE_EMBEDDING_FILE)
        with open(image_file_path, 'w') as f:
            f.write('')
        self.assertEqual(self.e_gen._get_image_embeddings_file(), image_file_path)

    def test_get_set_of_embedding_names(self):
        embeddings = [['name1', 0.1, 0.2], ['name2', 0.3, 0.4]]
        expected_set = {'name1', 'name2'}
        self.assertEqual(self.e_gen._get_set_of_embedding_names(embeddings), expected_set)

    def test_get_embedding(self):
        embedding_file_path = os.path.join(self.temp_dir, "embeddings.tsv")
        with open(embedding_file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['name', 'dim1', 'dim2'])  # header
            writer.writerow(['name1', '0.1', '0.2'])
            writer.writerow(['name2', '0.3', '0.4'])
        read_embeddings = self.e_gen._get_embedding(embedding_file_path)
        expected_embeddings = [['name1', '0.1', '0.2'], ['name2', '0.3', '0.4']]
        self.assertEqual(read_embeddings, expected_embeddings)

    def test_get_dimensions(self):
        self.assertEqual(self.e_gen.get_dimensions(), 1024)

    def test_get_ppi_embeddings_file_with_filename(self):
        test_embedding_filename = "test_ppi_embedding_file.txt"
        expected_file_path = os.path.join(self.temp_dir, test_embedding_filename)
        with open(expected_file_path, 'w') as f:
            f.write('')
        e_gen = EmbeddingGenerator(ppi_embeddingdir=self.temp_dir, image_embeddingdir=self.temp_dir,
                                   embedding_filenames=[test_embedding_filename, test_embedding_filename])
        result_file_path = e_gen._get_ppi_embeddings_file()

        self.assertEqual(result_file_path, expected_file_path)
        self.assertIn(test_embedding_filename, e_gen._embedding_names)

    def test_get_image_embeddings_file_with_filename(self):
        test_embedding_filename = "test_ppi_embedding_file.txt"
        expected_file_path = os.path.join(self.temp_dir, test_embedding_filename)
        with open(expected_file_path, 'w') as f:
            f.write('')
        e_gen = EmbeddingGenerator(ppi_embeddingdir=self.temp_dir, image_embeddingdir=self.temp_dir,
                                   embedding_filenames=[test_embedding_filename, test_embedding_filename])
        result_file_path = e_gen._get_image_embeddings_file()

        self.assertEqual(result_file_path, expected_file_path)
        self.assertIn(test_embedding_filename, e_gen._embedding_names)

    def test_get_image_embeddings(self):
        image_embedding_file_name = "image_embeddings.tsv"
        e_gen = EmbeddingGenerator(image_embeddingdir=self.temp_dir,
                                   embedding_filenames=[None, image_embedding_file_name])

        image_file_path = os.path.join(self.temp_dir, image_embedding_file_name)
        with open(image_file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['name', 'dim1', 'dim2'])  # header
            writer.writerow(['image1', '0.1', '0.2'])
            writer.writerow(['image2', '0.3', '0.4'])

        expected_embeddings = [['image1', '0.1', '0.2'], ['image2', '0.3', '0.4']]
        image_embeddings = e_gen._get_image_embeddings()
        self.assertEqual(image_embeddings, expected_embeddings)

    @patch('cellmaps_coembedding.muse_sc.muse_fit_predict')
    @patch.object(MuseCoEmbeddingGenerator, '_get_image_embeddings')
    @patch.object(MuseCoEmbeddingGenerator, '_get_ppi_embeddings')
    def test_get_next_embedding(self, mock_get_ppi_embeddings, mock_get_image_embeddings, mock_muse_fit_predict):
        generator = MuseCoEmbeddingGenerator(outdir=self.temp_dir, jackknife_percent=1.0)

        def mock_ppi_embeddings():
            generator._embedding_names.append('PPI')
            return [['name1', 0.1, 0.2], ['name2', 0.3, 0.4]]

        def mock_image_embeddings():
            generator._embedding_names.append('image')
            return [['name2', 0.5, 0.6], ['name3', 0.7, 0.8]]

        mock_get_ppi_embeddings.side_effect = mock_ppi_embeddings
        mock_get_image_embeddings.side_effect = mock_image_embeddings

        mock_embeddings = np.array([[0.9, 0.8]])
        mock_muse_fit_predict.return_value = (None, mock_embeddings)

        result_embeddings = list(generator.get_next_embedding())

        self.assertEqual(len(result_embeddings), len(mock_embeddings))

        expected_file_path = os.path.join(self.temp_dir, 'muse_test_genes.txt')
        self.assertTrue(os.path.exists(expected_file_path), "Test genes file was not created.")

        with open(expected_file_path, 'r') as file:
            lines = file.read().splitlines()

        expected_lines = ['name2']
        self.assertEqual(lines, expected_lines, "File content does not match expected gene names.")
