#! /usr/bin/python

import sys
import numpy as np
import tensorflow as tf
"""Return list of operations from TFLite model
"""


def get_model_ops(tflite_file):
    intr = tf.lite.Interpreter(tflite_file)
    intr.allocate_tensors()
    ops = intr._get_ops_details()
    return ops


"""Return list of tensors from TFLite model
"""


def get_model_tensors(tflite_file):
    intr = tf.lite.Interpreter(tflite_file)
    intr.allocate_tensors()
    tensors = intr.get_tensor_details()
    return tensors


"""Generate binary adjacency matrix from a tflite model. The adjacency matrix is symmetric and 
undirected.
"""


def generate_adj_matrix(tflite_file):
    intr = tf.lite.Interpreter(tflite_file)
    intr.allocate_tensors()
    ops = intr._get_ops_details()
    adj_mat = np.zeros((len(ops), len(ops)), dtype=int)
    for i in range(len(ops) - 1):
        for j in range(i + 1, len(ops)):
            if i != j:
                if len(set(ops[i]['outputs']).intersection(set(ops[j]['inputs']))) > 0:
                    adj_mat[i][j] = 1
                    adj_mat[j][i] = 1
                if len(set(ops[i]['inputs']).intersection(set(ops[j]['outputs']))) > 0:
                    adj_mat[i][j] = 1
                    adj_mat[j][i] = 1
    return adj_mat


"""Generate directed acyclic graph (DAG) from a tflite model.
"""


def generate_dag(tflite_file):
    intr = tf.lite.Interpreter(tflite_file)
    intr.allocate_tensors()
    ops = intr._get_ops_details()
    adj_mat = np.zeros((len(ops), len(ops)), dtype=int)
    for i in range(len(ops) - 1):
        for j in range(i + 1, len(ops)):
            if i != j:
                if len(set(ops[i]['outputs']).intersection(set(ops[j]['inputs']))) > 0:
                    adj_mat[i][j] = 1
                if len(set(ops[i]['inputs']).intersection(set(ops[j]['outputs']))) > 0:
                    adj_mat[j][i] = 1
    return adj_mat


"""Generate Compressed Sparse Row format (CSR) of a adjacency matrix. Details on CSR are given at
https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format).
"""


def get_csr(adj_matrix):
    row_ptr = []
    col_ind = []
    assert (adj_matrix.shape[0] == adj_matrix.shape[1])
    n = adj_matrix.shape[0]
    cnt = 0
    for i in range(n):
        first = True
        for j in range(n):
            if adj_matrix[i][j] == 1:
                col_ind.append(j)
                if first == True:
                    first = False
                    row_ptr.append(cnt)
                cnt += 1
    row_ptr.append(cnt)
    return row_ptr, col_ind


"""Perform basic spectral clustering given a tflite model. The graph in this case is symmetric, undirected with
unit weight per edge. Therefore, the spectral clustering is performed on a binary (0-1) adjacency matrix derived 
from the tflite model.
"""


def spectral_cluster(tflite_file):
    adj_matrix = generate_adj_matrix(tflite_file)
    L = np.diag(np.sum(adj_matrix, axis=0)) - adj_matrix
    e_val, e_vec = np.linalg.eig(L)
    vecs = e_vec[:, np.argsort(e_val)]
    return vecs.T[1]
