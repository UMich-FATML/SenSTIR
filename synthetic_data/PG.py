import sys
sys.path.append("..")
sys.path.append("../Fair-PGRank/")

import numpy as np
import math

from models import CustomLinearModel

from progressbar import progressbar

import fair_training_ranking
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from train_yahoo_dataset import on_policy_training
from YahooDataReader import YahooDataReader
import torch
from models import NNModel, LinearModel
from evaluation import evaluate_model

from sklearn.linear_model import LogisticRegression

num_queries = 100
cs_size = 10

relevances = np.load('data/relevance.npy')
_X = np.load('data/X.npy')
X = np.zeros((num_queries*cs_size,3))
X[:, 1:3] = _X
majority_status_train = np.load('data/majority_status.npy')
X[:,0] = ~majority_status_train

rels = [relevances[i*cs_size:(i+1)*cs_size] for i in range(num_queries)]
feats = [X[i*cs_size:(i+1)*cs_size,:] for i in range(num_queries)]
group_identities_train = [X[i*cs_size:(i+1)*cs_size,0] for i in range(num_queries)]

test_relevances = np.load('data/relevance_test.npy')
_X_test = np.load('data/X_test.npy')
X_test = np.zeros((num_queries*cs_size,3))
X_test[:, 1:3] = _X_test
majority_status_test = np.load('data/majority_status_test.npy')
X_test[:,0] = ~majority_status_test

rels_test = [test_relevances[i*cs_size:(i+1)*cs_size] for i in range(num_queries)]
feats_test = [X_test[i*cs_size:(i+1)*cs_size,:] for i in range(num_queries)]
group_identities_test = [X_test[i*cs_size:(i+1)*cs_size,0] for i in range(num_queries)]

dr = YahooDataReader(None)
dr.data = (feats, rels)
vdr = dr

dr_test = YahooDataReader(None)
dr_test.data = (feats_test, rels_test)
vdr_test = dr_test

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
args = Namespace(conditional_model=True, gpu_id=None, progressbar=True, evaluate_interval=250, input_dim=3,
                 eval_rank_limit=1000,
                fairness_version="asym_disparity", entropy_regularizer=0.0, save_checkpoints=False, num_cores=1,
                pooling='concat_avg', dropout=0.0, hidden_layer=8, group_feat_id=0, summary_writing=False,
                 group_fairness_version="asym_disparity",early_stopping=False, lr_scheduler=False,
                 validation_deterministic=False, evalk=1000, reward_type="ndcg", baseline_type="value",
                 use_baseline=True, entreg_decay=0.0, skip_zero_relevance=True, eval_temperature=1.0, optimizer="Adam",
                clamp=False)
torch.set_num_threads(args.num_cores)
args.group_feat_id = 0
args.progressbar = False

args.lr = 0.1
args.lr_scheduler = True
args.weight_decay = 0.0
args.lr_decay = 0.5
args.group_identities = group_identities_train

model_params_list = []
disparities = []
lambdas_list = [25]
for lgroup in lambdas_list:
    torch.set_num_threads(args.num_cores)
    args.epochs = 5
    args.progressbar = False
    args.weight_decay = 0.0
    args.sample_size = 10
    args.pooling = False
    args.skip_zero_relevance = True
    args.validation_deterministic = False
    args.lambda_reward = 1.0
    args.lambda_ind_fairness = 0.0
    args.lambda_group_fairness = lgroup


    model = CustomLinearModel(D=args.input_dim, use_bias=False, fix_weight_dim=0)

    model = on_policy_training(dr, vdr, model, args=args)
    results = evaluate_model(model, vdr_test, fairness_evaluation=True, group_fairness_evaluation=True,
                             deterministic=False, args=args, num_sample_per_query=10)
    print('stochastic test', results)

    results = evaluate_model(model, vdr_test, fairness_evaluation=True, group_fairness_evaluation=True,
                             deterministic=True, args=args, num_sample_per_query=10)
    print('deterministic test', results)

    model_params_list.append(model.w.weight.data.tolist()[0])
    print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
    disparities.append(results["avg_group_asym_disparity"])

weights = np.array([np.array(model.w.weight.data.tolist()[0], dtype = 'float32').reshape(-1,1), np.array([0.], dtype='float32')])

tf.reset_default_graph()
num_docs_per_query = 10
num_queries = 100
LR = LogisticRegression(C = 100).fit(_X, majority_status_train)
sens_directions = LR.coef_
weights, train_logits, test_logits, _  = fair_training_ranking.train_fair_nn(_X,
                                                        relevances,
                                                        majority_status_train,
                                                        X_test = _X_test,
                                                        relevance_test = test_relevances,
                                                        group_membership_test = majority_status_test,
                                                        num_items_per_query = num_docs_per_query,
                                                        tf_prefix='sensei',
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
                                                        num_monte_carlo_samples = 10,
                                                        bias = True,
                                                        init_range = .0001,
                                                        entropy_regularizer = .0,
                                                        baseline_ndcg = True
                                                        )
