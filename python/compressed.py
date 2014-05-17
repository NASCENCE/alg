__author__ = 'Marijn Stollenga, Jan Koutnik'

import copy
import numpy as np
from scipy.fftpack import dct, idct


class Compressor(object):
    def __init__(self, matrices, matrix_importance = None, dim_importance = None, max_index = 10e5):
        if type(matrices) is not type([]):
            raise Exception("provide a LIST of matrices")

        if not matrix_importance:
            matrix_importance = []
            for i, m in enumerate(matrices):
                matrix_importance.append(np.log(m.size))

        if not dim_importance:
            dim_importance = []
            for i, m in enumerate(matrices):
                dim_importance.append(np.log(np.array(m.shape).astype(np.float32)) + 1)

        self.matrices = matrices
        self.matrix_importance = np.array(matrix_importance)
        self.dim_importance = np.array(dim_importance)

        self.coordinates = []
        for i, m in enumerate(self.matrices):
            self.coordinates.append(create_coordinates(m, dim_importance[i], max_index))

    def split_values(self, values):
        assert(len(values) > len(self.matrices))
        values_per_matrix = []

        counter = np.ones(len(self.matrices))
        for i in range(len(self.matrices)):
            values_per_matrix.append([values[i]])

        for v in values[len(self.matrix_importance):]:
            index = np.argmax(self.matrix_importance / counter)
            values_per_matrix[index].append(v)
            counter[index] += 1
        return values_per_matrix

    def decode(self, values, bias = 0.0):
        # values = np.exp(values)
        values_per_matrix = self.split_values(values)

        for i in range(len(self.matrices)):
            self.matrices[i].fill(0)

            for index, v in enumerate(values_per_matrix[i]):
                if index == 0:
                    self.matrices[i][self.coordinates[i][index]] = v + bias
                else:
                    self.matrices[i][self.coordinates[i][index]] = v

            if len(self.matrices[i].shape) == 1:
                self.matrices[i] = idct1d(self.matrices[i])
            if len(self.matrices[i].shape) == 2:
                self.matrices[i] = idct2d(self.matrices[i])
            if len(self.matrices[i].shape) == 3:
                self.matrices[i] = idct3d(self.matrices[i])
        # print self.matrices
        # sys.exit(1)
        return self.matrices

def idct3d(a):
    raise "not supported"

def idct2d(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def idct1d(a):
    return idct(a, norm='ortho')

def comp(d, min_map, max_map):
    def compare(c1, c2):
        l1, l2 = np.sum(c1), np.sum(c2)

        wl1, wl2 = np.sum(c1 / d), np.sum(c2 / d)
        if wl1 < wl2: return -1
        if wl1 > wl2: return 1

        min_coord, max_coord = min_map[l1], max_map[l1]

        s1, s2 = None, None
        for i in range(len(c1)):
            ts1, ts2 = min(c1[i] - min_coord[i], max_coord[i] - c1[i]), min(c2[i] - min_coord[i], max_coord[i] - c2[i])
            s1 = min(s1, ts1) if s1 else ts1
            s2 = min(s2, ts2) if s2 else ts2

        if s1 < s2: return -1
        if s1 > s2: return 1
        return 0
    return compare

def recurse(a, index, coord, max_index):
    if len(a.shape) == index:
        return [coord]

    coordinates = []

    for i in range(a.shape[index]):
        if max_index and i > max_index: break
        new_coord = copy.copy(coord)
        new_coord.append(i)
        coordinates.extend(recurse(a, index+1, new_coord, max_index))
    return coordinates

def create_coordinates(a, d, max_index = None):
    coordinates = []
    for c in recurse(a, 0, [], max_index):
       coordinates.append(tuple(c))

    min_map, max_map = {}, {}
    for c in coordinates:
        l = np.sum(c)
        if l not in min_map:
            min_map[l], max_map[l] = c, c
        min_map[l] = np.minimum(c, min_map[l])
        max_map[l] = np.maximum(c, max_map[l])

    coordinates.sort(comp(d, min_map, max_map))

    return coordinates

