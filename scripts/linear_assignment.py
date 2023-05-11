'''
code for computing linear assignments using lapjv
'''

import numpy as np
import unittest
from numpy import array, dstack, float32, float64, linspace, meshgrid, random, sqrt
from scipy.spatial.distance import cdist
from lapjv import lapjv


def base_solve(W, max_dummy_cost_value=1000):
    '''
    Gives hungarian solve for a non-square matrix. it's roughly from:

    NOTE: this ** MINIMIZES COST **. So, if you're handing sims, make sure to negate them!

    https://github.com/jmhessel/multi-retrieval/blob/master/bipartite_utils.py

    returns i_s, j_s, cost such that:
    for i, j in zip(i_s, j_s)

    are the (i, j) row column entries selected.

    cost is sum( cost[i, j] for i, j in zip(i_s, j_s) )

    '''
    if np.sum(np.abs(W)) > max_dummy_cost_value:
        print('Warning, you values in your matrix may be too big, please raise max_dummy_cost_value')


    orig_shape = W.shape
    if orig_shape[0] != orig_shape[1]:
        if orig_shape[0] > orig_shape[1]:
            pad_idxs = [[0, 0], [0, W.shape[0]-W.shape[1]]]
            col_pad = True
        else:
            pad_idxs = [[0, W.shape[1]-W.shape[0]], [0, 0]]
            col_pad = False
        W = np.pad(W, pad_idxs, 'constant', constant_values=max_dummy_cost_value)

    sol, _, cost = lapjv(W)

    i_s = np.arange(len(sol))
    j_s = sol[i_s]

    sort_idxs = np.argsort(-W[i_s, j_s])
    i_s, j_s = map(lambda x: x[sort_idxs], [i_s, j_s])

    if orig_shape[0] != orig_shape[1]:
        if col_pad:
            valid_idxs = np.where(j_s < orig_shape[1])[0]
        else:
            valid_idxs = np.where(i_s < orig_shape[0])[0]
        i_s, j_s = i_s[valid_idxs], j_s[valid_idxs]

    m_cost = 0.0
    for i, j in zip(i_s, j_s):
        m_cost += W[i, j]

    return i_s, j_s, m_cost


# unit tests from https://github.com/src-d/lapjv/blob/master/test.py . except the last one which is non-square.
class LapjvTests(unittest.TestCase):
    def test_basic(self):
        arr = -np.array([[1.0, 1.0],
                         [1.5, 1.0],
                         [3.0, 2.6]])
        # should be 1.5 and 2.6
        i_s, j_s, _ = base_solve(arr)

        assert set(zip(i_s, j_s)) == set([(1, 0), (2,1)])


    def _test_random_100(self, dtype):
        random.seed(777)
        size = 100
        dots = random.random((size, 2))
        grid = dstack(meshgrid(linspace(0, 1, int(sqrt(size))),
                               linspace(0, 1, int(sqrt(size))))).reshape(-1, 2)
        cost = cdist(dots, grid, "sqeuclidean").astype(dtype)
        cost *= 100000 / cost.max()
        row_ind_lapjv, col_ind_lapjv, _ = base_solve(cost)
        # Obtained from pyLAPJV on Python 2.7
        row_ind_original = array([
            32, 51, 99, 77, 62, 1, 35, 69, 57, 42, 13, 24, 96, 26, 82, 52, 65,
            6, 95, 7, 63, 47, 28, 45, 74,
            61, 34, 14, 94, 31, 25, 3, 71, 49, 58, 83, 91, 93, 23, 98, 36, 40,
            4, 97, 21, 92, 89, 90, 29, 46,
            79, 2, 76, 84, 72, 64, 33, 37, 41, 15, 59, 85, 70, 78, 81, 20, 18,
            30, 8, 66, 38, 87, 44, 67, 68,
            39, 86, 54, 11, 50, 16, 17, 56, 0, 5, 80, 10, 48, 60, 73, 53, 75,
            55, 19, 22, 12, 9, 88, 43, 27])

        # we have to do this conversion to get to the (r, c) format...
        A = np.zeros((100, 100))
        for i in range(100):
            A[i, row_ind_original[i]] = 1

        row_ind_original = np.arange(A.shape[0])
        col_ind_original = np.argmax(A, axis=1)

        # make sure the set of index pairs is the same
        orig_pairs = set(zip(row_ind_original, col_ind_original))
        new_pairs = set(zip(row_ind_lapjv, col_ind_lapjv))
        assert orig_pairs == new_pairs, (orig_pairs, new_pairs)


    def test_random_100_float64(self):
        self._test_random_100(np.float64)

    def test_random_100_float32(self):
        self._test_random_100(np.float32)

    def test_1024(self):
        random.seed(777)
        size = 1024
        dots = random.random((size, 2))
        grid = dstack(meshgrid(linspace(0, 1, int(sqrt(size))),
                               linspace(0, 1, int(sqrt(size))))).reshape(-1, 2)
        cost = cdist(dots, grid, "sqeuclidean")
        cost *= 100000 / cost.max()
        row_ind_lapjv32, col_ind_lapjv32, _ = base_solve(cost)
        self.assertEqual(len(set(col_ind_lapjv32)), dots.shape[0])
        self.assertEqual(len(set(row_ind_lapjv32)), dots.shape[0])
        row_ind_lapjv64, col_ind_lapjv64, _ = base_solve(cost)

        f32_pairs = set(zip(row_ind_lapjv32, col_ind_lapjv32))
        f64_pairs = set(zip(row_ind_lapjv64, col_ind_lapjv64))
        assert f32_pairs == f64_pairs


if __name__ == '__main__':
    unittest.main()
