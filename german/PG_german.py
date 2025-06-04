import sys
sys.path.append('..')
sys.path.append('Fair-PGRank/')

import numpy as np
import math

from progressbar import progressbar

from train_yahoo_dataset import on_policy_training
from YahooDataReader import YahooDataReader
import torch
from models import NNModel, LinearModel
from evaluation import evaluate_model
import pandas as pd
import pickle as pkl

import numpy as np
import fair_training_ranking
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from optparse import OptionParser
from get_german import get_data

def parse_args():

    parser = OptionParser()

    # SenSR parameters
    parser.add_option("--lamb", type="float", dest="lamb")
    parser.add_option("--epochs", type="int", dest="epochs")
    parser.add_option("--seed", type="int", dest="seed")

    (options, args) = parser.parse_args()

    return options

def main():
    options = parse_args()
    print(options)

    lamb = options.lamb
    epochs = options.epochs
    seed = options.seed
    sens_directions = get_data(seed = seed, gender = False)
    # read data

    dr = YahooDataReader(None)
    X, relevances, male, age = pd.read_pickle(r'german_train_rank_{}.pkl'.format(str(seed)))

    cs_size = 10
    num_queries = int(X.shape[0] / cs_size)
    rels = [relevances[i*cs_size:(i+1)*cs_size] for i in range(num_queries)]
    feats = [X[i*cs_size:(i+1)*cs_size,:] for i in range(num_queries)]
    age = [age[i*cs_size:(i+1)*cs_size] for i in range(num_queries)]

    dr.data = (feats,rels)
    vdr = YahooDataReader(None)

    _X_test, relevances_test, male_test, age_test= pd.read_pickle(r'german_test_rank_{}.pkl'.format(str(seed)))

    num_queries = int(_X_test.shape[0] / cs_size)

    rels_test = [relevances_test[i*cs_size:(i+1)*cs_size] for i in range(num_queries)]
    feats_test = [_X_test[i*cs_size:(i+1)*cs_size,:] for i in range(num_queries)]
    age_test = [age_test[i*cs_size:(i+1)*cs_size] for i in range(num_queries)]

    vdr.data = (feats_test,rels_test)

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = Namespace(conditional_model=True, gpu_id=None, progressbar=True, evaluate_interval=250, input_dim=29,
                     eval_rank_limit=1000,
                    fairness_version="asym_disparity", entropy_regularizer=0.0, save_checkpoints=False, num_cores=1,
                    pooling='concat_avg', dropout=0.0, hidden_layer=8, summary_writing=False,
                     group_fairness_version="asym_disparity",early_stopping=False, lr_scheduler=False,
                     validation_deterministic=False, evalk=1000, reward_type="ndcg", baseline_type="value",
                     use_baseline=True, entreg_decay=0.0, skip_zero_relevance=True, eval_temperature=1.0, optimizer="Adam",
                    clamp=False)
    torch.set_num_threads(args.num_cores)
    args.progressbar = False

    args.group_feat_id = 3

    torch.set_num_threads(args.num_cores)
    args.lambda_reward = 1.0
    args.lambda_ind_fairness = 0.0
    args.lambda_group_fairness = lamb

    args.lr = 0.001
    args.epochs = epochs
    args.progressbar = False
    args.weight_decay = 0.0
    args.sample_size = 25
    args.optimizer = "Adam"
    args.group_identities = age

    model = LinearModel(D=args.input_dim)

    model = on_policy_training(dr, vdr, model, args=args)

    weights = np.array([np.array(model.w.weight.data.tolist()[0], dtype = 'float32').reshape(-1,1), np.array(model.w.bias.data.tolist(), dtype='float32')])

    tf.reset_default_graph()

    X_queries, relevances, male_sex, age= pd.read_pickle(r'german_train_rank_{}.pkl'.format(str(seed)))
    X_queries_test, relevances_test, male_sex_test, age_test = pd.read_pickle(r'german_test_rank_{}.pkl'.format(str(seed)))

    CF_X_train = np.copy(X_queries)
    CF_X_train[np.where(X_queries[:,3] == 1), 3] = 0
    CF_X_train[np.where(X_queries[:,3] == 1), 4] = 1
    CF_X_train[np.where(X_queries[:,4] == 1), 4] = 0
    CF_X_train[np.where(X_queries[:,4] == 1), 3] = 1

    CF_X_test = np.copy(X_queries_test)
    CF_X_test[np.where(X_queries_test[:,3] == 1), 3] = 0
    CF_X_test[np.where(X_queries_test[:,3] == 1), 4] = 1
    CF_X_test[np.where(X_queries_test[:,4] == 1), 4] = 0
    CF_X_test[np.where(X_queries_test[:,4] == 1), 3] = 1

    _  = fair_training_ranking.train_fair_nn(X_queries,
                                                            relevances,
                                                            age,
                                                            CF_X_train = CF_X_train,
                                                            CF_X_test = CF_X_test,
                                                            X_test = X_queries_test,
                                                            relevance_test = relevances_test,
                                                            group_membership_test = age_test,
                                                            num_items_per_query = 10,
                                                            tf_prefix='PG_'+str(lamb) +'_'+ str(epochs),
                                                            weights=weights,
                                                            n_units = [],
                                                            lr=0.04,
                                                            fair_start=1.,
                                                            batch_size=1,
                                                            epoch=0,
                                                            verbose=True,
                                                            activ_f = tf.nn.relu,
                                                            l2_reg=0.0,
                                                            plot=True,
                                                            sens_directions=sens_directions,
                                                            seed=None,
                                                            simul=False,
                                                            num_monte_carlo_samples = 25,
                                                            bias = True,
                                                            init_range = .0001,
                                                            entropy_regularizer = .0,
                                                            baseline_ndcg = True
                                                            )





    #         print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
    #         plt_data_pl[i+1] = [results["ndcg"], results["avg_group_asym_disparity"]]

if __name__ == '__main__':
    main()
