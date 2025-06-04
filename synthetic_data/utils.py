import numpy as np
from sklearn.preprocessing import OneHotEncoder

def generate_synthetic_LTR_data(majority_proportion = .8, num_queries = 100, num_docs_per_query = 10, seed=0):
    num_items = num_queries*num_docs_per_query
    X = np.random.uniform(0,3, size = (num_items,2))
    relevance = X[:,0] + X[:,1]

    # i don't know why but the "fair policy" paper clips the values between 0 and 5
    relevance = np.clip(relevance, 0.0,5.0)
    majority_status = np.random.choice([True, False], size=num_items, p=[majority_proportion, 1-majority_proportion])
    X[~majority_status, 1] = 0
    return X, relevance, majority_status
