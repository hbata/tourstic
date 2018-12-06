from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import os
import click


def load_data():
    with open('dataset/features_mat.pkl', 'rb') as f:
        features = pkl.load(f)
    with open('dataset/labels_mat.pkl', 'rb') as f:
        labels = pkl.load(f)
    return features, labels


def run_training_kfold(n_splits, kernel='linear', save_clf=True):
    X, Y = load_data()
    stand_X = StandardScaler()
    X_s = stand_X.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    svm_clf = svm.SVC(C=10.0, kernel=kernel, probability=True, class_weight='balanced', verbose=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train_idx, test_idx in cv.split(X_s, Y):
        probas = svm_clf.fit(X_s[train_idx], Y[train_idx]).predict_proba(X_s[test_idx])
        fpr, tpr, threshold = roc_curve(Y[test_idx], probas[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.4, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    if save_clf:
        if not os.path.exists('model/'):
            os.makedirs('model')
        with open('model/svm_slf.pkl', 'wb') as f:
            pkl.dump(svm_clf, f)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    auc_std = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, auc_std),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('auc_kfolds')
    plt.show()


# this is to add as click command
def main():
    kernel = 'linear'
    n_splits = 6
    run_training_kfold(n_splits, kernel)
    with open('model/svm_slf.pkl', 'rb') as f:
        model = pkl.load(f)


if __name__ == '__main__':
    main()
