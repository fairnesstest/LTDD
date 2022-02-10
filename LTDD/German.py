import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics

import sys
sys.path.append(os.path.abspath('..'))

from scipy import stats
## Load dataset
dataset_orig = pd.read_csv('../dataset/GermanData.csv')

## Drop categorical features
dataset_orig = dataset_orig.drop(['1','2','4','5','8','10','11','12','14','15','16','17','18','19','20'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()

## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])

dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3, dataset_orig['credit_history'])

dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])

## ADD Columns
dataset_orig['credit_history=Delay'] = 0
dataset_orig['credit_history=None/Paid'] = 0
dataset_orig['credit_history=Other'] = 0

dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1, dataset_orig['credit_history=Delay'])
dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1, dataset_orig['credit_history=None/Paid'])
dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1, dataset_orig['credit_history=Other'])

dataset_orig['savings=500+'] = 0
dataset_orig['savings=<500'] = 0
dataset_orig['savings=Unknown/None'] = 0

dataset_orig['savings=500+'] = np.where(dataset_orig['savings'] == 1, 1, dataset_orig['savings=500+'])
dataset_orig['savings=<500'] = np.where(dataset_orig['savings'] == 2, 1, dataset_orig['savings=<500'])
dataset_orig['savings=Unknown/None'] = np.where(dataset_orig['savings'] == 3, 1, dataset_orig['savings=Unknown/None'])

dataset_orig['employment=1-4 years'] = 0
dataset_orig['employment=4+ years'] = 0
dataset_orig['employment=Unemployed'] = 0

dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1, dataset_orig['employment=1-4 years'])
dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1, dataset_orig['employment=4+ years'])
dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1, dataset_orig['employment=Unemployed'])

dataset_orig = dataset_orig.drop(['credit_history','savings','employment'],axis=1)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

np.random.seed(4)

aod_b = []
eod_b = []
aod_a = []
eod_a = []
acc_a = []
acc_b = []
recall_a = []
recall_b = []
false_a = []
false_b = []
DI_a = []
SPD_a = []
DI_b = []
SPD_b = []
ce_list = []
ce_times = []

for k in range(100):
    print('------the {}th turn------'.format(k))
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.15, random_state=None,shuffle = True)

    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']
    column_train = [column for column in X_train]

    from sklearn.neural_network import MLPClassifier

    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.metrics import ClassificationMetric
    from aif360.metrics import BinaryLabelDatasetMetric

    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                   unfavorable_label=0.0,
                                   df=dataset_orig_test,
                                   label_names=['Probability'],
                                   protected_attribute_names=['sex'],
                                   )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dataset_pred = dataset_t.copy()
    dataset_pred.labels = y_pred
    attr = dataset_t.protected_attribute_names[0]
    idx = dataset_t.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    print("Disparate Impact:", class_metrics.disparate_impact())
    print("Statistical Parity Difference:", class_metrics.statistical_parity_difference())
    DI_b.append(class_metrics.disparate_impact())
    SPD_b.append(class_metrics.statistical_parity_difference())

    print("accuracy :", class_metrics.accuracy())
    print("recall :", class_metrics.recall())
    print("far:", class_metrics.false_positive_rate())
    print("aod:", class_metrics.average_odds_difference())
    print("eod:", class_metrics.equal_opportunity_difference())
    acc_b.append(class_metrics.accuracy())
    recall_b.append(class_metrics.recall())
    false_b.append(class_metrics.false_positive_rate())
    aod_b.append(class_metrics.average_odds_difference())
    eod_b.append(class_metrics.equal_opportunity_difference())

    slope_store = []
    intercept_store = []
    rvalue_store = []
    pvalue_store = []
    column_u = []
    flag = 0
    ce = []
    times = 0
    def Linear_regression(x, slope, intercept):
        return x * slope + intercept
    for i in column_train:
        flag = flag + 1
        if i != 'sex':
            slope,intercept,rvalue,pvalue,stderr=stats.linregress(X_train['sex'], X_train[i])
            rvalue_store.append(rvalue)
            pvalue_store.append(pvalue)
            if pvalue < 0.05:
                times = times + 1
                column_u.append(i)
                ce.append(flag)
                slope_store.append(slope)
                intercept_store.append(intercept)
                X_train[i] = X_train[i] - Linear_regression(X_train['sex'], slope, intercept)

    ce_times.append(times)
    ce_list.append(ce)
    X_train = X_train.drop(['sex'],axis = 1)

    for i in range(len(column_u)):
        X_test[column_u[i]] = X_test[column_u[i]] - Linear_regression(X_test['sex'], slope_store[i], intercept_store[i])

    X_test = X_test.drop(['sex'],axis = 1)

    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.metrics import ClassificationMetric
    from aif360.metrics import BinaryLabelDatasetMetric

    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                   unfavorable_label=0.0,
                                   df=dataset_orig_test,
                                   label_names=['Probability'],
                                   protected_attribute_names=['sex'],
                                   )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dataset_pred = dataset_t.copy()
    dataset_pred.labels = y_pred
    attr = dataset_t.protected_attribute_names[0]
    idx = dataset_t.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    print("Disparate Impact:", class_metrics.disparate_impact())
    print("Statistical Parity Difference:", class_metrics.statistical_parity_difference())
    DI_a.append(class_metrics.disparate_impact())
    SPD_a.append(class_metrics.statistical_parity_difference())

    print("accuracy :", class_metrics.accuracy())
    print("recall :", class_metrics.recall())
    print("far:", class_metrics.false_positive_rate())
    print("aod:", class_metrics.average_odds_difference())
    print("eod:", class_metrics.equal_opportunity_difference())
    acc_a.append(class_metrics.accuracy())
    recall_a.append(class_metrics.recall())
    false_a.append(class_metrics.false_positive_rate())
    aod_a.append(class_metrics.average_odds_difference())
    eod_a.append(class_metrics.equal_opportunity_difference())

#DI,SPD are the original value, different from the value reported in the paper.
print('---Original---')
print('Aod before:', np.mean(np.abs(aod_b)))
print('Eod before:', np.mean(np.abs(eod_b)))
print('Acc before:', np.mean(acc_b))
print('Far before:', np.mean(false_b))
print('recall before:', np.mean(recall_b))
print('DI before:', np.mean(DI_b))
print('SPD before:', np.mean(np.abs(SPD_b)))

print('---LTDD---')
print('Aod after:', np.mean(np.abs(aod_a)))
print('Eod after:', np.mean(np.abs(eod_a)))
print('Acc after:', np.mean(acc_a))
print('Far after:', np.mean(false_a))
print('recall after:', np.mean(recall_a))
print('DI after:', np.mean(DI_a))
print('SPD after:', np.mean(np.abs(SPD_a)))

