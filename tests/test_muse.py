import unittest
from unittest.mock import patch

import numpy as np

from cellmaps_coembedding.muse_sc import make_matrix_from_labels, write_result_to_file, train_model, muse_fit_predict


class TestMuse(unittest.TestCase):
    def test_make_matrix_from_labels(self):
        labels = np.array([0, 0, 1, 1, 2])
        expected_matrix = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        result_matrix = make_matrix_from_labels(labels)
        np.testing.assert_array_equal(result_matrix, expected_matrix)

    @patch('csv.writer')
    def test_write_result_to_file(self, mock_writer):
        # create temporary directory
        # write to it
        # check if content as expected
        # delete temporary directory
        pass


if __name__ == '__main__':
    unittest.main()
