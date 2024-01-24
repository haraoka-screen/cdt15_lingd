import os
import sys
import numpy as np
import pytest

from lingd import LiNGD


@pytest.fixture
def init():
    return lambda :np.random.seed(0)

def _generate_data(B, e_vars=None, sample_size=1000):
    if e_vars is None:
        e_vars = np.ones(B.shape[1])

    es = []
    a = (3 * e_vars) ** 0.5
    for i in range(len(e_vars)):
        es.append(np.random.uniform(-a[i], a[i], size=sample_size))
    es = np.array(es).T
    
    # X = Ae
    A = np.linalg.inv(np.eye(B.shape[0]) - B)
    X = (A @ es.T).T
    
    return X

@pytest.fixture
def test_data():
    np.random.seed(0)

    B = np.array([
        [ 0.0, 0.0, 0.0, 0.0, 0.0,],
        [ 1.2, 0.0, 0.0,-0.3, 0.0,],
        [ 0.0, 2.0, 0.0, 0.0, 0.0,],
        [ 0.0, 0.0,-1.0, 0.0, 0.0,],
        [ 0.0, 3.0, 0.0, 0.0, 0.0,],
    ])
    X = _generate_data(B)

    return X

@pytest.fixture
def test_data2():
    np.random.seed(0)

    B = np.array([
        [ 0.0, 0.5],
        [ 0.5, 0.0],
    ])
    X = _generate_data(B)

    return X

def test_fit_success(init, test_data, test_data2):
    init()

    X = test_data

    k = 10
    model = LiNGD(k=k)
    model.fit(X)

    assert len(model.adjacency_matrices_) == k

    for adj in model.adjacency_matrices_:
        assert adj.shape == (X.shape[1], X.shape[1])

    assert sum(np.diff(model.costs_) >= 0) == k - 1

    assert sum(model.is_stables_) > 0

    # special case
    X = test_data2

    k = 10
    model = LiNGD(k=k)
    model.fit(X)

    assert len(model.adjacency_matrices_) == 1

def test_fit_exception(init, test_data):
    init()

    X = test_data

    # invalid range
    try:
        model = LiNGD(k=0)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid type
    try:
        model = LiNGD(k=1.0)
        model.fit(X)
    except:
        pass
    else:
        raise AssertionError

def test_properties_exception(init):
    init()

    model = LiNGD()

    try:
        model.estimates_causal_effects(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    try:
        model.adjacency_matrices_
    except RuntimeError:
        pass
    else:
        raise AssertionError

    try:
        model.costs_
    except RuntimeError:
        pass
    else:
        raise AssertionError

    try:
        model.is_stables_
    except RuntimeError:
        pass
    else:
        raise AssertionError

def test_estimates_causal_effects_success(init, test_data):
    init()

    X = test_data

    k = 10
    model = LiNGD(k=k)
    model.fit(X)
    effects = model.estimates_causal_effects(1)

    assert len(effects) == k

    for effect in effects:
       assert len(effect) == X.shape[1]

def test_estimates_causal_effects_exception(init, test_data):
    init()

    X = test_data

    model = LiNGD()
    model.fit(X)

    try:
        effects = model.estimates_causal_effects(-1)
    except ValueError:
        pass
    else:
        raise AssertionError
