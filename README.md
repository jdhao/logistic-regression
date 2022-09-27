# Description

This is my implementation for Logistic regression for a classification task,
dropout during training is also included.
Dataset used in training and evaluation is [breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

The model training is done using SGD (stochastic gradient descent).

You can check the derivation of derivative for weight in `doc.pdf`.

# Dependency

Run the following command to install dependencies:

```
pip install -r requirements.txt
```

# How to run

The complete code is in `run_sgd.py`:

```
python run_sgd.py
```
