# this code is largely taken from Ashudeep Singh and found here: https://github.com/ashudeep/Fair-PGRank/blob/master/GermanCredit/German%20Credit%20Data%20Preprocessing.ipynb

import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeCV

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def get_weights(Y, ratio_relevant):
    num_relevant = np.sum(Y == 1)
    num_not_relevant = len(Y) - num_relevant
    y = (1-ratio_relevant) / num_not_relevant
    x = ratio_relevant / num_relevant
    ratios_col = Y * x + (1-Y)* y
    return ratios_col

def get_age_direction(X_queries, X_queries_test):

    X = np.delete(X_queries, [0], axis=1)
    X_test = np.delete(X_queries_test, [0], axis=1)

    ridge = RidgeCV().fit(X, X_queries[:,0])
    print('R^2 train', ridge.score(X,X_queries[:,0]))
    print('R^2 test', ridge.score(X_test,X_queries_test[:,0]))

    # removing duplicates
    X = unique_rows(X_queries)
    X_test = unique_rows(X_queries_test)

    ridge = RidgeCV().fit(X[:, 1:], X[:,0])

    print('R^2 train', ridge.score(X[:, 1:], X[:,0]))
    print('R^2 test', ridge.score(X_test[:, 1:], X_test[:,0]))

    w = ridge.coef_
    print(w)
    direction_1 = np.zeros(X_queries.shape[1])
    direction_1[1:] = w
    direction_2 = np.eye(X_queries.shape[1])[:,0]

    sens_directions = np.zeros((2,X_queries.shape[1]))
    sens_directions[0,:] = direction_1
    sens_directions[1,:] = direction_2

    return sens_directions

def get_gender_direction(X_queries, X_queries_test, male_sex, male_sex_test):
    # gender
    X = np.delete(X_queries, [3,4], axis=1)
    LR = LogisticRegressionCV(Cs = [.0001, .001, .01, .1, 1], class_weight = 'balanced').fit(X, male_sex)
    w = LR.coef_[0]
    print(w)
    direction_1 = np.zeros(X_queries.shape[1])
    direction_2 = np.eye(X_queries.shape[1])[:,3]
    direction_3 = np.eye(X_queries.shape[1])[:,4]

    direction_1[0:3] = w[0:3]
    direction_1[5:X_queries.shape[1]] = w[5-2:X_queries.shape[1]-2]

    sens_directions = np.zeros((3,X_queries.shape[1]))
    sens_directions[0,:] = direction_1
    sens_directions[1,:] = direction_2
    sens_directions[2,:] = direction_3

    X = np.delete(X_queries, [3,4], axis=1)
    X_test = np.delete(X_queries_test, [3,4], axis=1)
    print('test acc', LR.score(X_test, male_sex_test))
    print('train acc', LR.score(X, male_sex))
    CM = confusion_matrix(male_sex_test, np.argmax(LR.predict_proba(X_test), axis = 1))

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    FNR = FN / (FN + TP)
    FPR = FP / (FP+TN)
    print('TNR test', 1-FPR)
    print('TPR test', 1-FNR)

    CM = confusion_matrix(male_sex, np.argmax(LR.predict_proba(X), axis = 1))

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    FNR = FN / (FN + TP)
    FPR = FP / (FP+TN)
    print('TNR train', 1-FPR)
    print('TPR train', 1-FNR)
    return sens_directions

def get_data(seed = 0, gender = True):
    np.random.seed(seed)
    try:
        data_X, data_Y, gender_identities_train, age_identities_train = pkl.load(open("german_train_rank_{}.pkl".format(str(seed)), "rb"))
        test_X, test_Y, gender_identities_test, age_identities_test = pkl.load(open("german_test_rank_{}.pkl".format(str(seed)), "rb"))
    except (OSError, IOError) as e:
        df = pd.read_csv("german_credit_data.csv", index_col=0)

        df = df.fillna(value="NA")
        df['age_binary'] = df.apply(lambda row: 1 if row['Age'] >= 25 else 0, axis=1)

        preprocess = make_column_transformer(
            (StandardScaler(), ['Age', 'Credit amount', 'Duration']),
            (OneHotEncoder(sparse=False), ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk', 'age_binary'])
        )

        mat = preprocess.fit_transform(df)
        np.random.shuffle(mat)

        X = mat[:, :-4]
        Y = mat[:, -1-2]
        age = mat[:, -1]

        num_feats = X.shape[1]
        numX = X.shape[0]

        datasize = 500 #number of queries in training
        query_size = 10 # number of items per query
        split_on_doc = 0.8 #seperates items for training/testing
        testsize = 100 # number of queries in testing
        ratio_relevant = .4 # out of the 10 items only 40% are relevant
        ratios_col = Y * ratio_relevant + (1-Y)*(1-ratio_relevant)

        # generate a candidate set of size 10 everytime
        data_X = np.zeros((datasize*query_size, num_feats))
        data_Y = np.zeros(datasize*query_size)

        test_X = np.zeros((testsize*query_size, num_feats))
        test_Y = np.zeros(testsize*query_size)

        gender_identities_train = np.zeros(datasize*query_size)
        gender_identities_test = np.zeros(testsize*query_size)

        age_identities_train = np.zeros(datasize*query_size)
        age_identities_test = np.zeros(testsize*query_size)

        print("Sampling between 0 and {} for train".format(numX*split_on_doc))

        p = get_weights(Y[:int(numX*split_on_doc)], ratio_relevant)
        #p = ratios_col[0:int(numX*split_on_doc)]
        #p = p / np.sum(p)

        for i in range(datasize):
            cs_indices = np.random.choice(np.arange(0, numX*split_on_doc, dtype=int), size=query_size, p=p)
            while np.sum(Y[cs_indices]) == 0:
                cs_indices = np.random.choice(np.arange(0, numX*split_on_doc, dtype=int), size=query_size, p=p)
            data_X[i*query_size:(i+1)*query_size, :] = X[cs_indices, :]
            data_Y[i*query_size:(i+1)*query_size] = Y[cs_indices]
            gender_identities_train[i*query_size:(i+1)*query_size] = X[cs_indices,4]
            age_identities_train[i*query_size:(i+1)*query_size] = age[cs_indices]

        print("Sampling between {} and {} for test".format(numX*split_on_doc, numX))
        p = get_weights(Y[int(numX*split_on_doc):], ratio_relevant)
        #p = ratios_col[int(numX*split_on_doc):]
        #p = p/sum(p)

        for i in range(testsize):
            cs_indices = np.random.choice(np.arange(0, numX*(1-split_on_doc), dtype=int), size=query_size, p=p)
            while np.sum(Y[int(numX*split_on_doc) + cs_indices]) == 0:
                cs_indices = np.random.choice(np.arange(0, numX*(1-split_on_doc), dtype=int), size=query_size, p=p)
            test_X[i*query_size:(i+1)*query_size, :] = X[int(numX*split_on_doc) + cs_indices, :]
            test_Y[i*query_size:(i+1)*query_size] = Y[int(numX*split_on_doc) + cs_indices]
            gender_identities_test[i*query_size:(i+1)*query_size] = X[int(numX*split_on_doc) + cs_indices,4]
            age_identities_test[i*query_size:(i+1)*query_size] = age[int(numX*split_on_doc) + cs_indices]

        pkl.dump((data_X, data_Y, gender_identities_train, age_identities_train), open("german_train_rank_{}.pkl".format(str(seed)), "wb"))

        pkl.dump((test_X, test_Y, gender_identities_test, age_identities_test), open("german_test_rank_{}.pkl".format(str(seed)), "wb"))

    if gender:
        return get_gender_direction(data_X, test_X, gender_identities_train, gender_identities_test)
    else:
        return get_age_direction(data_X, test_X)
