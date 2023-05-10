#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_coembedding` package."""


import unittest
from cellmaps_coembedding.runner import CellmapsCoEmbedder
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError


class TestCellmapsCoEmbeddingRunner(unittest.TestCase):
    """Tests for `cellmaps_coembedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsCoEmbedder(outdir='foo')

        self.assertIsNotNone(myobj)

    def test_constructor_no_outdir(self):
        """ Tests run()"""
        try:
            myobj = CellmapsCoEmbedder()
            self.fail('Expected exception')
        except CellmapsCoEmbeddingError as ce:
            self.assertEqual('outdir is None', str(ce))
