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
                                                               '--apms_embedding',
                                                               'apms',
                                                               '--image_embedding',
                                                               'image'])

        self.assertEqual(res.verbose, 0)
        self.assertEqual(res.logconf, None)

        someargs = ['-vv', '--logconf', 'hi',
                    'foo',
                    '--apms_embedding',
                    'apms',
                    '--image_embedding',
                    'image']
        res = cellmaps_coembeddingcmd._parse_arguments('hi', someargs)

        self.assertEqual(res.verbose, 2)
        self.assertEqual(res.logconf, 'hi')

    def test_main(self):
        """Tests main function"""

        # try where loading config is successful
        try:
            temp_dir = tempfile.mkdtemp()
            res = cellmaps_coembeddingcmd.main(['myprog.py',
                                                 'foo',
                                                  '--apms_embedding',
                                                  'apms',
                                                  '--image_embedding',
                                                  'image'])
            self.assertEqual(res, 2)
        finally:
            shutil.rmtree(temp_dir)
