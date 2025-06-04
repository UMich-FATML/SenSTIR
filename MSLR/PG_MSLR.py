import sys
sys.path.append('Fair-PGRank/')
sys.path.append('../Fair-PGRank/')
sys.path.append('..')


import numpy as np
import math
import fair_training_ranking

from train_yahoo_dataset import on_policy_training
from YahooDataReader import YahooDataReader
import torch
from models import NNModel, LinearModel

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from optparse import OptionParser
from sklearn.linear_model import RidgeCV


def get_sensitive_direction(X_queries_train, X_queries_testate, X_queries_test):
    X_train = np.delete(X_queries_train, [135-4], axis=1)
    X_test = np.delete(X_queries_testate, [135-4], axis=1)
    X_test = np.delete(X_queries_test, [135-4], axis=1)

    ridge = RidgeCV().fit(X_train, X_queries_train[:,135-4])
    print('train R^2', ridge.score(X_train, X_queries_train[:,135-4]))
    print('testate R^2', ridge.score(X_test, X_queries_testate[:,135-4]))
    print('test R^2', ridge.score(X_test, X_queries_test[:,135-4]))

    w = ridge.coef_

    direction_1 = np.zeros(X_queries_train.shape[1])
    direction_2 = np.eye(X_queries_train.shape[1])[:,135-4]

    direction_1[0:135-4] = w[0:135-4]
    direction_1[135-4 + 1:] = w[135-4:]

    sens_directions = np.zeros((2,X_queries_train.shape[1]))
    sens_directions[0,:] = direction_1
    sens_directions[1,:] = direction_2
    return sens_directions

def parse_args():

    parser = OptionParser()

    # SenSR parameters
    parser.add_option("--lamb", type="float", dest="lamb")
    parser.add_option("--l2_reg", type="float", dest="l2_reg")
    parser.add_option("--n_units", type="int", dest="n_units")

    (options, args) = parser.parse_args()

    return options

def main():
    options = parse_args()
    print(options)

    lamb = options.lamb
    l2_reg = options.l2_reg
    n_units = options.n_units

    if n_units == 0:
        n_units = []
    else:
        n_units = [n_units]

    X_train = np.load('data/X_train.npy')
    rel_train = np.load('data/y_train.npy')
    group_train = np.load('data/group_train.npy')

    X_test = np.load('data/X_test.npy')
    rel_test = np.load('data/y_test.npy')
    group_test = np.load('data/group_test.npy')

    sens_directions = get_sensitive_direction(X_train, X_test, X_test)

    lr = .001
    epoch = 40*1700
    batch_size = 10
    num_monte_carlo_samples = 32
    entropy_regularizer = 0.
    num_docs_per_query = 20

    _  = fair_training_ranking.train_fair_nn(X_train, rel_train, group_train, X_test = X_test, relevance_test = rel_test, group_membership_test = group_test,
                        num_items_per_query = num_docs_per_query, sens_directions = sens_directions, tf_prefix='PG_'+str(lamb) + '_' + str(l2_reg),
                        n_units = n_units,
                        lr=lr,
                        batch_size=batch_size,
                        epoch=epoch,
                        verbose=True,
                        activ_f = tf.nn.relu,
                        l2_reg=l2_reg,
                        plot=True,
                        fair_reg=0.,
                        fair_start=1.0,
                        seed=None,
                        simul=False,
                        num_monte_carlo_samples = num_monte_carlo_samples,
                        bias = True,
                        init_range = .0001,
                        entropy_regularizer = entropy_regularizer,
                        baseline_ndcg = True,
                        PG = True,
                        PG_reg = lamb
                        )

    #         print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
    #         plt_data_pl[i+1] = [results["ndcg"], results["avg_group_asym_disparity"]]

if __name__ == '__main__':
    main()
