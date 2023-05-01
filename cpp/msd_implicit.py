import argparse
import codecs
import logging
import time
from scipy import sparse
import numpy as np
import pandas as pd
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "faiss_als": FaissAlternatingLeastSquares,
    "tfidf": TFIDFRecommender,
    "cosine": CosineRecommender,
    "bpr": BayesianPersonalizedRanking,
    "lmf": LogisticMatrixFactorization,
    "bm25": BM25Recommender,
}

def get_model(model_name):
    print(f"getting model {model_name}")
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError(f"Unknown Model '{model_name}'")

    # some default params
    if model_name.endswith("als"):
        params = {"factors": 128, "dtype": np.float32}
    elif model_name == "bm25":
        params = {"K1": 100, "B": 0.5}
    elif model_name == "bpr":
        params = {"factors": 63}
    elif model_name == "lmf":
        params = {"factors": 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)

def msd_load(filename):
    utility = np.loadtxt(filename, dtype=float, delimiter=',')
    users = utility[:, 0]
    items = utility[:, 1]
    ratings = utility[:, 2]
    rating_count = len(ratings)
    user_count = int(np.max(users)) + 1
    item_count = int(np.max(items)) + 1
    print(f'{user_count} users, {item_count} items, {rating_count} ratings')
    uirs = sparse.csr_matrix(
        (ratings, (users, items)), shape=(user_count, item_count), dtype=float)
    iurs = uirs.T
    return iurs

def msd_lmf(iurs):
    model_name = "lmf"
    model = get_model(model_name)
    uirs = iurs.tocsr().T.tocsr()
    model.fit(uirs)
    msd_save_embeddings(model, 'lmf_')
    return model

def msd_als(iurs):
    model_name = "als"
    model = get_model(model_name)
    logging.debug("weighting matrix by bm25_weight")
    iurs = bm25_weight(iurs, K1=100, B=0.8)
    model.approximate_similar_items = False
    uirs = iurs.tocsr().T.tocsr()
    model.fit(uirs)
    msd_save_embeddings(model, 'als_')
    return model

def msd_save_embeddings(model, prefix):
    user_factors = model.user_factors
    item_factors = model.item_factors
    d_user_factors = len(user_factors[0])
    d_item_factors = len(item_factors[0])
    users = pd.DataFrame(user_factors)
    users.columns = [f'd{i}' for i in range(d_user_factors)]
    users.index.name = 'User'
    items = pd.DataFrame(item_factors)
    items.columns = [f'd{i}' for i in range(d_item_factors)]
    items.index.name = 'Item'
    print(f'saving {prefix}user_factors.csv ...')
    users.to_csv(f'{prefix}user_factors.csv')
    print(f'saving {prefix}item_factors.csv ...')
    items.to_csv(f'{prefix}item_factors.csv')
    return model

if __name__ == "__main__":

    iurs = msd_load("canon_train.csv")
    msd_als(iurs)
    msd_lmf(iurs)
