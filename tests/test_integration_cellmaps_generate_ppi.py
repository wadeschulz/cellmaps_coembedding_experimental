#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration Tests for `cellmaps_generate_ppi` package."""

import os

import unittest
from cellmaps_generate_ppi import cellmaps_generate_ppicmd

SKIP_REASON = 'CELLMAPS_GENERATE_PPI_INTEGRATION_TEST ' \
              'environment variable not set, cannot run integration ' \
              'tests'

@unittest.skipUnless(os.getenv('CELLMAPS_GENERATE_PPI_INTEGRATION_TEST') is not None, SKIP_REASON)
class TestIntegrationCellmaps_generate_ppi(unittest.TestCase):
    """Tests for `cellmaps_generate_ppi` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_something(self):
        """Tests parse arguments"""
        self.assertEqual(1, 1)
