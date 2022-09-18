## Logistic Regression

One of the most popular parametric linear models is [logistic regression](https://www.coursera.org/lecture/machine-learning/classification-wlPeP). Although designed for regression problems with the output of continuous values in [0, 1], [a little trick](https://machinelearningmastery.com/logistic-regression-for-machine-learning/) (interpreting continuous values as posterior class probabilities) makes it one of the most useful tools available to CV/ML engineers for building a strong [baseline](https://www.coursera.org/lecture/machine-learning/classification-wlPeP) before delving into deep learning.

### Problem definition

Let’s assume we are interested in using logistic regression to classify a set of observations into two classes ([binary classification](https://en.wikipedia.org/wiki/Binary_classification)), f.ex - if an email is spam or not. For this exercise we use the [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). You can easily load and use this dataset from the scikit-learn python package as [follows](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

For our coding challenge, we are interested in learning parameters of a logistic regression model on the Breast Cancer Dataset. Alongside this doc you’d find our bare-bone implementation of logistic regression. In our implementation, we intend to train our model with [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) + [log-loss](https://www.quora.com/What-is-an-intuitive-explanation-for-the-log-loss-function) and get comparable performance to its better known public [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).

To test the correctness of your implementation we use publicly available [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) as a strong baseline to provide guidelines on the expected accuracy. `NOTE`: We don’t expect your implementation to outperform [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) (we won’t complain if it does :smile:).

### Deliverable

#### Part 1

  - Modify the arguments for `SGDClassifier` to fully support a linear model.
  - These modifications would depend on your implementation of `__compute_loss`

#### Part 2

  - Derive gradient updates for weights and bias value for the model from scratch
  - You can do it on a sheet of paper and send us a photo

#### Part 3

As a coding task we’d like you to implement the following function(s)

1. `clean_data`
  - Remove any noise in the training data with heuristics
2. `fit`
  - Given features and ground truth labels
  - Loop of data for epocs / iterations
  - Build random mini-batch
  - Compute log-loss using `__compute_loss` below
  - Compute gradients given log-loss, mini-batch
  - Updated the weight (`self.w`) and bias (`self.b`) of the model
3. `predict`
  - Given samples predict the labels with trained weights and bias
  - Currently we have set predict to assign random labels
4. `__compute_loss`
  - Compute loss over a batch
  - Currently we set it to hard coded value (0.0)
5. `__compute_gradient`
  - Compute gradient give loss/batch
  - Currently set as zero (No updates)

#### How will we evaluate your code

1. Correctness : Does your code do the right thing
2. Objective: Comment on your choice of loss function
3. Convergence: Does the Loss decrease with the number of iterations
4. Blind baseline: Is your classifier better than a random classifier

### Report

1. Is minimizing the loss the best criteria to perform early stopping ?
2. Does the model guarantee performance on an unseen dataset ?
3. How does `lr` and `batch_size` affect convergence ?
4. Bonus question :

  a. Is it possible to modify the training data and learn just the weight vector ?

  b. Add a function `__dropout`, which randomly sets some of feature values to zero during training. How will you incorporate it during fit / predict ?

  c. Does `__dropout`  help in convergence / overfitting ?
