#! /usr/bin/env python

import logging


logger = logging.getLogger(__name__)


class CellmapsgenerateppiRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsgenerateppiRunner.run` method
        :type int:
        """
        logger.debug('In constructor')

    def run(self):
        """
        Runs CM4AI Generate PPI


        :return:
        """
        logger.debug('In run method')
        return 0
