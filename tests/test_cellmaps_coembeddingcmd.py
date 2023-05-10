#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_coembedding` package."""

import os
import tempfile
import shutil

import unittest
from cellmaps_coembedding import cellmaps_coembeddingcmd


class TestCellmapsCoEmbedding(unittest.TestCase):
    """Tests for `cellmaps_coembedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_parse_arguments(self):
        """Tests parse arguments"""
        res = cellmaps_coembeddingcmd._parse_arguments('hi', ['foo',
                                                               '--ppi_embeddingdir',
                                                               'apms',
                                                               '--image_embeddingdir',
                                                               'image',
                                                               '--image_downloaddir',
                                                               'download'])

        self.assertEqual(res.verbose, 0)
        self.assertEqual(res.logconf, None)

        someargs = ['-vv', '--logconf', 'hi',
                    'foo',
                    '--ppi_embeddingdir',
                    'apms',
                    '--image_embeddingdir',
                    'image', '--image_downloaddir', 'download']
        res = cellmaps_coembeddingcmd._parse_arguments('hi', someargs)

        self.assertEqual(2, res.verbose)
        self.assertEqual('hi', res.logconf)
        self.assertEqual('foo', res.outdir)
        self.assertEqual('apms', res.ppi_embeddingdir)
        self.assertEqual('image', res.image_embeddingdir)
        self.assertEqual('download', res.image_downloaddir)

    def test_main(self):
        """Tests main function"""

        # try where loading config is successful
        try:
            temp_dir = tempfile.mkdtemp()
            res = cellmaps_coembeddingcmd.main(['myprog.py',
                                                 'foo',
                                                  '--ppi_embeddingdir',
                                                  'apms',
                                                  '--image_embeddingdir',
                                                  'image',
                                                 '--image_downloaddir',
                                                'download'])
            self.assertEqual(2, res)
        finally:
            shutil.rmtree(temp_dir)
