import typing
from functools import reduce

import numpy as np
import gudhi as gd
from ripser import Rips
from diagram2vec import persistence_curve
from persim import PersistenceImager
from hodgelaplacians import HodgeLaplacians
from tqdm.auto import tqdm

from dataload import Record


def get_simplicial_complex_skeleton(distance_matrix, max_edge_length, max_simplex_dim):
    skeleton = gd.RipsComplex(
        distance_matrix=distance_matrix, 
        max_edge_length=max_edge_length,
    )
    simplex_tree = skeleton.create_simplex_tree(max_dimension=max_simplex_dim) 
    return simplex_tree.get_skeleton(max_simplex_dim)


def build_hodge_spectrum_dataset_for_dim(records: typing.List[Record], laplacian_dim: int, n_eigvals: int, smallest: bool = True):
    X, y, indices = [], [], []
    for idx, record in enumerate(tqdm(records)):
        distance_matrix = 1 - record.data.corr(method='pearson').to_numpy()
        dists = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]
        skeleton = get_simplicial_complex_skeleton(distance_matrix, max_edge_length=np.quantile(dists, 0.01), max_simplex_dim=laplacian_dim + 1)
        skeleton = list(skeleton)

        if laplacian_dim != 0 and len(skeleton) > 3 * 0.01 * distance_matrix.shape[0] ** (laplacian_dim + 1):
            continue
        hl = HodgeLaplacians(skeleton, maxdimension=laplacian_dim + 1, mode='gudhi')
        L = hl.getHodgeLaplacian(laplacian_dim)

        eigvals = np.linalg.eigvalsh(L.todense())
        if n_eigvals is not None:
            if smallest:
                eigvals = eigvals[:n_eigvals]
            else:  # largest
                eigvals = eigvals[-n_eigvals:]
                
        X.append(eigvals)
        y.append(int(record.patient))
        indices.append(idx)
    return np.array(X, dtype=object), np.array(y), indices


def get_hodge_spectrum_dataset(records: typing.List[Record], laplacian_dims: typing.List[int], n_eigvals: typing.List[int] = None, smallest: bool = True):
    X, y, indices = [], [], []
    for i, dim in enumerate(laplacian_dims):
        n_eigvals_for_dim = None if n_eigvals is None else n_eigvals[i]
        X_for_dim, y_for_dim, indices_for_dim = build_hodge_spectrum_dataset_for_dim(records, dim, n_eigvals_for_dim, smallest)

        num_of_eigenvals = min([x.shape[0] for x in X_for_dim])
        X_for_dim = np.array([x[:num_of_eigenvals] for x in X_for_dim])

        X.append(X_for_dim)
        y.append(y_for_dim)
        indices.append(indices_for_dim)

    indices_intersection = reduce(np.intersect1d, indices)

    XX, yy = [], []
    for i, inds in enumerate(indices):
        isin = np.isin(inds, indices_intersection)
        XX.append(X[i][isin])
        yy.append(y[i][isin])

    X = np.hstack(XX)
    y = yy[0]

    return X, y


def get_persistence_diagrams_dataset(records: typing.List[Record]):
    X_diagrams, y_diagrams = [], []
    for record in tqdm(records):
        distance_matrix = 1 - record.data.corr(method='pearson').to_numpy()
        np.fill_diagonal(distance_matrix, 0)
        diagram = Rips(verbose=False).fit_transform(distance_matrix, distance_matrix=True)
        X_diagrams.append(diagram)
        y_diagrams.append(int(record.patient))
    return np.array(X_diagrams, dtype=object), np.array(y_diagrams)


def get_betti_curves_from_persistence_diagrams(persistence_diagrams, flatten_dims: bool = True, **kwargs):
    params = {'m': 500, 'f': 'linear', 'b': 0.01}
    params.update(kwargs)
    betti_curves = persistence_curve(persistence_diagrams, **params)
    if flatten_dims:
        betti_curves = np.array([np.hstack(x) for x in betti_curves])
    return betti_curves


def get_betti_curves_dataset(records: typing.List[Record], flatten_dims: bool = True, **kwargs):
    X, y = get_persistence_diagrams_dataset(records)
    return get_betti_curves_from_persistence_diagrams(X, flatten_dims, **kwargs), y


def get_persistence_images_from_persistence_diagrams(persistence_diagrams, flatten_images: bool = True, **kwargs):
    params = {'pixel_size': 0.015}
    params.update(kwargs)

    pimgr = PersistenceImager(**params)
    pimages = np.array(pimgr.fit_transform([np.concatenate((x[0][:-1], x[1]), axis=0) for x in persistence_diagrams], skew=True))

    if flatten_images:
        pimages = pimages.reshape(pimages.shape[0], -1)

    return pimages


def get_persistence_images_dataset(records: typing.List[Record], flatten_images: bool = True, **kwargs):
    X, y = get_persistence_diagrams_dataset(records)
    return get_persistence_images_from_persistence_diagrams(X, flatten_images, **kwargs), y
