import random
from collections import Counter
from ts2vec import TS2Vec
# LOAD SVARA-FORM DATASET
import os
from sklearn.metrics import f1_score

# Choosing confidence metric
import numpy as np
import pandas as pd

import numpy as np
from sklearn.calibration import calibration_curve

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV



experiment_name = 'pretrain-varnam_finetune-bhairavi'
params_path = f"params/{experiment_name}.json"


def load_pkl(path):
    import pickle
    file = open(path,'rb')
    return pickle.load(file)


def prepare_data(series_list):
    import numpy as np
    N = len(series_list)
    F = series_list[0].shape[1]
    Tmax = max(seq.shape[0] for seq in series_list)

    X = np.full((N, Tmax, F), np.nan, dtype=np.float32)
    for i, seq in enumerate(series_list):
        T = seq.shape[0]
        X[i, :T, :] = seq

    return X


print("Loading parameters")
with open(params_path, 'r') as f:
    import json
    params = json.load(f)

train_path = params['train_path']
test_path = params['test_path']


results_path = f"runs/{experiment_name}/results.csv"

print("Loading model")
## Single model
model = TS2Vec(
    input_dims=params['input_dims'],
    device='cpu',
    output_dims=params["output_dims"],
    hidden_dims=params["hidden_dims"],
    depth=params["depth"],
    lr=params["lr"],
    batch_size=params["batch_size"])


model.load(params['model_path'])

SVARAS = ['S', 'R', 'G', 'M', 'P', 'D', 'N']
print("Loading data")
all_svaras = load_pkl(train_path)
all_svaras = [x for x in all_svaras if x['fold']=='validation']
series_list = [x['curr'][0] for x in all_svaras]
y_train = [SVARAS.index(x['curr'][1]) for x in all_svaras]
svara_names = set(y_train)

X_train = prepare_data(series_list)

all_svaras = load_pkl(test_path)
all_svaras = [x for x in all_svaras if x['fold']=='test']
series_list = [x['curr'][0] for x in all_svaras]
y_test = [SVARAS.index(x['curr'][1]) for x in all_svaras]
svara_names = set(y_test)

X_test = prepare_data(series_list)

print("Encoding")
X_train = model.encode(X_train, encoding_window='full_series')  # n_instances x output_dims
X_test = model.encode(X_test, encoding_window='full_series')  # n_instances x output_dims

print("Training model")
def compute_multiclass_ece(y_probs, y_true, n_bins=10):
    """
    Computes multi-class Expected Calibration Error (ECE).

    Parameters:
        y_true: True class labels (shape: [n_samples])
        y_probs: Predicted probabilities (shape: [n_samples, n_classes])
        n_bins: Number of bins for calibration

    Returns:
        Average ECE across all classes
    """
    n_classes = y_probs.shape[1]
    ece_list = []

    for c in range(n_classes):
        y_true_binary = (y_true == c).astype(int)  # One-vs-All for class c

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true_binary, y_probs[:, c], n_bins=n_bins, strategy='uniform')

        # Compute histogram bins for actual bin counts
        bin_counts, _ = np.histogram(y_probs[:, c], bins=n_bins, range=(0, 1))

        # Ensure all arrays have the same length
        min_bins = min(len(bin_counts), len(prob_true), len(prob_pred))
        bin_counts = bin_counts[:min_bins]
        prob_true = prob_true[:min_bins]
        prob_pred = prob_pred[:min_bins]

        # Compute ECE for this class
        ece = np.sum((bin_counts / len(y_true)) * np.abs(prob_true - prob_pred))
        ece_list.append(ece)

    return np.mean(ece_list)  # Average ECE across all classes


# Define ranges of hyperparameters for Random Forest
n_estimators_range = [50, 100, 200]
max_depth_range = [None, 10, 20]
min_samples_split_range = [2, 5]
min_samples_leaf_range = [1, 2]
max_features_range = ['auto', 'sqrt', 'log2']

results = []
# Iterate over all possible hyperparameter combinations
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for min_samples_split in min_samples_split_range:
            for min_samples_leaf in min_samples_leaf_range:
                    for scale in [True, False]:
                        for calib in [True, False]:
                            if scale:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                            else:
                                X_train_scaled = X_train
                                X_test_scaled = X_test

                            print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}, "
                                  f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

                            # Initialize Random Forest Classifier with the current set of hyperparameters
                            rf = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42
                            )

                            if calib:
                                rf = CalibratedClassifierCV(rf, method="isotonic", cv=3)

                            # Train the model
                            rf.fit(X_train_scaled, y_train)

                            # Evaluate accuracy
                            t1 = time.time()
                            y_pred = rf.predict(X_train_scaled)
                            t2 = time.time()

                            # Assuming you have trained a GPC model as 'rf' with your training data
                            train_confidences = rf.predict_proba(X_train_scaled)  # Confidence scores
                            train_preds = rf.predict(X_train_scaled)

                            test_confidences = rf.predict_proba(X_test_scaled)  # Confidence scores
                            test_preds = rf.predict(X_test_scaled)

                            train_score = f1_score(train_preds, np.array(y_train), average='macro')
                            test_score = f1_score(test_preds, np.array(y_test), average='macro')

                            ece_train = compute_multiclass_ece(train_confidences, np.array(y_train), n_bins=10)
                            ece_test = compute_multiclass_ece(test_confidences, np.array(y_test), n_bins=10)

                            results.append((n_estimators, max_depth, min_samples_split, min_samples_leaf, scale, calib, train_score, test_score, ece_train, ece_test, (t2-t1)/len(X_train_scaled)))

df = pd.DataFrame(results, columns=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'scale', 'calib', 'train_f1', 'test_f1', 'ece_train', 'ece_test', 'compute_time'])
df['method'] = "Random Forest"
df.to_csv(results_path, index=False)
