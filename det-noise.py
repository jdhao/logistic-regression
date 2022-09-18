import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def main():
    x, y = load_breast_cancer(return_X_y=True)
    x = StandardScaler().fit_transform(x)

    # local outlier factor
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    y_pred = clf.fit_predict(x)
    x_scores = clf.negative_outlier_factor_

    outlier_num = (y_pred == -1).sum()
    idx = np.where(y_pred == -1)[0]

    print(f"outlier num: {outlier_num}")
    print(f"LOF, outlier idx: {idx}, score: {x_scores[idx]}")

    # isolation forest
    clf2 = IsolationForest(max_samples='auto', contamination=0.02)
    clf2.fit(x)

    y_pred = clf2.predict(x)
    idx = np.where(y_pred == -1)[0]
    print(f"Isolation forest, outlier idx: {idx}")


if __name__ == "__main__":
    main()

# Reference
# https://scikit-learn.org/stable/modules/outlier_detection.html
