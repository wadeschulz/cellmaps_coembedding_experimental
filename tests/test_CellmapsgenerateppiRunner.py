#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_generate_ppi` package."""


import unittest
from cellmaps_generate_ppi.runner import CellmapsgenerateppiRunner
from cellmaps_generate_ppi.exceptions import CellmapsgenerateppiError


class TestCellmapsgenerateppirunner(unittest.TestCase):
    """Tests for `cellmaps_generate_ppi` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsgenerateppiRunner()

        self.assertIsNotNone(myobj)

    def test_run_no_outdir(self):
        """ Tests run()"""
        myobj = CellmapsgenerateppiRunner()
        try:
            myobj.run()
            self.fail('Expected exception')
        except CellmapsgenerateppiError as ce:
            self.assertEqual('outdir must be set', str(ce))
