import random
import csv
import numpy as np
import math
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def GenerateData(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])
    N = data.shape[0]
    X = data[:, :-1].astype(np.float64)
    X = normalize(X, axis=0, norm="max")  # avoid overflow problem
    X = np.insert(X, 0, values=1, axis=1)  # insert bias term
    print("Dimension of the feature data:", X.shape)
    y = data[:, -1].astype(np.int32).reshape(N, 1)
    test_id = np.random.choice(N, int(0.25 * N), replace=False)
    X_train = np.delete(X, test_id, axis=0)
    y_train = np.delete(y, test_id, axis=0)
    X_test = X[test_id]
    y_test = y[test_id]

    return X_train, y_train, X_test, y_test


def nabla_l(w_old, X, y):
    """
    A function expression for the gradient of l(w) w.r.t. w
    """

    p = 1 / (1 + math.e ** (-X @ w_old))
    return -X.T @ (y - p)


def nabla_2_l(w_old, X, y):
    """
    A function expression for the Hessian matrix of l(w) w.r.t. w
    """
    # %%
    # TODO: Compute the Hessian matrix of log-likelihood function
    p = 1 / (1 + math.e ** (-X @ w_old))
    q = 1 - 1 / (1 + math.e ** (-X @ w_old))
    return X.T @ ((p * q) * X)
    # End your code here
    # %%


def Newton_Solver_for_LR(w_old, nabla_l, nabla_2_l, X, y):
    """
    Solving the minimizer for log-likelihood in Logistic Regression((3.27) in the book).
        param nabla_l: function, nabla_l(w) is the gradient of l(w) w.r.t. w
        param nabla_2_l: function, nabla_2_l(w) is the Hessian matrix of l(w) w.r.t. w
        param X, y: parameters in l(w)
        return: optimize one_step for w_new
    """
    # %%
    # TODO: Finish the code here
    a = nabla_l(w_old, X, y)
    b = nabla_2_l(w_old, X, y)
    b = np.linalg.inv(b)
    w_new = w_old - b @ a
    # End your code here
    # %%

    return w_new


class LogisticRegression:
    def __init__(self, iter_num, Solver):
        self.name = "Logistic Regression"
        self.iter_num = iter_num
        self.Solver = Solver

    def train(self, X_train, y_train):
        self.w = np.random.randn(X_train.shape[1], 1)  # initialize w_0
        for _ in tqdm(range(self.iter_num)):
            self.w = self.Solver(
                w_old=self.w,
                nabla_l=nabla_l,
                nabla_2_l=nabla_2_l,
                X=X_train,
                y=y_train,
            )
        return self.w

    def predict(self, X_new):
        # %%
        # TODO: Finish the code here to predict the label for new data
        # Please return the probability vector P(X=1), rather than a 0-1 vector
        y_prob = 1 / (1 + math.e ** (-X_new @ self.w))
        # End your code here
        # %%

        return y_prob

    def evalute(self, X, y):
        # %%
        # TODO: Compute the accuracy when using parameter self.w
        cur = self.predict(X)
        idx = len(cur)
        cnt = 0
        for i in range(0, idx, 1):
            if cur[i] >= 0.5 and y[i] == 1:
                cnt += 1
            elif cur[i] < 0.5 and y[i] == 0:
                cnt += 1

        accuracy = cnt / len(y)
        # End your code here
        # %%

        return accuracy


def PlotROC(fpr, tpr):
    # %%
    # TODO: Plot the ROC
    plt.plot(fpr, tpr)
    plt.show()
    # %%


if __name__ == "__main__":
    set_seed(6)
    iter_num = 20
    data_path = "data.csv"

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = GenerateData(data_path)

    model = LogisticRegression(
        iter_num=iter_num,
        Solver=Newton_Solver_for_LR,
    )
    model.train(X_train=X_train, y_train=y_train)
    test_acc = model.evalute(X_test, y_test)
    print("Accuracy in test set: %.4f" % (test_acc))

    y_prob = model.predict(X_test)

    # %%
    # TODO: Compute the AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    auc_value = auc(fpr, tpr)
    # %%

    print("AUC in test set: %.4f" % (auc_value))

    PlotROC(fpr, tpr)
