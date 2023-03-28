#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_generate_ppi` package."""


import unittest
from cellmaps_generate_ppi.runner import CellmapsgenerateppiRunner


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

    def test_run(self):
        """ Tests run()"""
        myobj = CellmapsgenerateppiRunner()
        self.assertEqual(0, myobj.run())
