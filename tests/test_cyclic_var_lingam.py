import os
import sys
import itertools
import numpy as np
import networkx as nx
import pytest

from lingd import CyclicVARLiNGAM


@pytest.fixture
def init():
    return lambda :np.random.seed(0)

# value range of coefficients of adjacency matrices
COEF_RANGE = 0, 0.3

def _make_dag(n_features=8):
    sign = np.random.choice([-1, 1], size=(n_features, n_features))
    coef = np.random.uniform(*COEF_RANGE, size=(n_features, n_features))
    sparse = np.random.choice([0, 1], p=[0.3, 0.7], size=(n_features, n_features))

    graph = np.tril(sign * coef * sparse, k=-1)
    graph[np.isclose(graph, 0)] = 0
     
    return graph

def _make_tau_coefs(tau_len, n_features=8):
    coefs = []
    for i in range(tau_len):
        sign = np.random.choice([-1, 1], size=(n_features, n_features))
        coef = np.random.uniform(*COEF_RANGE, size=(n_features, n_features))
        sparse = np.random.choice([0, 1], p=[0.5, 0.5], size=(n_features, n_features))

        coef = sign * coef * sparse
        coef[np.isclose(coef, 0)] = 0
        
        coefs.append(coef)
    return coefs

def _generate_var_graph(n_features=4, lags=1, is_shuffle=True):
    B = np.empty((lags + 1, n_features, n_features))
    
    B[0] = _make_dag(n_features=n_features)
    B[1:] = _make_tau_coefs(lags, n_features=n_features)
    
    if is_shuffle:
        indices = np.random.permutation(n_features)
        for i in range(len(B)):
            B[i] = B[i][indices][:, indices]
        
    return B

def _make_var_data(B, T=100):
    k = len(B) - 1
    n_features = len(B[0])
    
    I_minus_B0_inv = np.linalg.pinv(np.eye(n_features) - B[0])

    X = np.empty((T * 2, n_features))

    for t in range(T * 2):
        lag = min(k, t)

        lag_data = np.array(X[t - lag:t])
        if len(lag_data) == 0:
            lag_data = 0
        else:
            lag_data = np.hstack(B[1:lag + 1]) @ np.hstack([*lag_data[::-1]]).reshape(-1, 1)

        e = np.random.uniform(size=(n_features, 1))
        X[t] = (I_minus_B0_inv @ (lag_data + e)).T

    return X[-T:]
    
def _make_cycle_path(B, max_trial=10):
    G = nx.from_numpy_array(B[0].T, create_using=nx.DiGraph)
    
    for _ in range(max_trial):
        # path to search
        pairs = list(itertools.permutations(range(len(B[0])), 2))
        pairs = np.random.permutation(pairs)

        target = None
        for s, d in pairs:
            if not nx.is_simple_path(G, (s, d)):
                continue

            # avoid to select a path which length is 1
            if True:
                paths = nx.shortest_simple_paths(G, s, d)
                path_lens = [len(p) - 1 for p in paths]
                if max(path_lens) < 2:
                    target = (s, d)
                    continue

            target = (s, d)
            break

        if target is not None:
            break
    
    if target is None:
        raise ValueError("no path")
        
    # add cyclic path
    Bc = B.copy()
    Bc[0][target[0], target[1]] = np.random.choice([-1, 1]) * np.random.uniform(*COEF_RANGE)
    
    return Bc

@pytest.fixture
def test_data():
    np.random.seed(0)

    B = _generate_var_graph(lags=2)
    B = _make_cycle_path(B)
    X = _make_var_data(B, T=1000)

    return X, B

def test_fit_success(init, test_data):
    init()

    X, B = test_data

    lags = 2 
    k = 5
    n_features = X.shape[1]
    n_bootstraps = 4

    # fit
    model = CyclicVARLiNGAM(lags=lags, k=k)
    model.fit(X)

    assert model.adjacency_matrices_.shape == (lags + 1, n_features, n_features)
    assert model.adjacency_matrices_list_.shape == (k, lags + 1, n_features, n_features)
    assert model.costs_.shape == (k, )
    assert model.is_stables_.shape == (k, )

    # boostrap
    model = CyclicVARLiNGAM(lags=lags, k=k)
    bs_result = model.bootstrap(X, n_bootstraps)

    assert np.array(bs_result.adjacency_matrices_).shape == (n_bootstraps, n_features, n_features*(lags + 1))

