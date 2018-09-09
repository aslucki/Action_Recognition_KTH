from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
import argparse
import numpy as np
import h5py


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True,
            help="Path to a file with features")
    ap.add_argument("--features_ds_name", required=True,
            help="Name of a dataset with features.")
    args = vars(ap.parse_args())

    np.random.seed(42)

    data = h5py.File(args['input_file'], 'r')
    features = data[args['features_ds_name']][:]
    labels = data['labels'][:]
    data.close()
    
    model = LinearSVC()
    params = {'C':[10**i for i in range(-5,5)]}
    grid_search_model  = GridSearchCV(model,
                                      param_grid=params,
                                      cv=5,
                                      return_train_score=True,
                                      verbose=3)
    
    grid_search_model.fit(features, labels)
    
    scores = cross_val_score(grid_search_model.best_estimator_,
                         features,
                         labels, cv=5)

    print("Mean accuracy: {}\n STD: {}".format(np.mean(scores), np.std(scores)))
