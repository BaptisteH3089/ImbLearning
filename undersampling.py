import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn import tree
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing as pre

chem_1 = ('/Users/baptistehessel/Documents/M2/class_raphael_sourty_DL/'
          'Projet_Imbalanced_Learning/creditcard.csv')

# Credit Card
df_1 = pd.read_csv(chem_1)


df_1.columns
# 28 variables, Amount et Time


# Décomposition en train et test
ind = list(df_1.index)
ind_test = np.random.choice(ind, math.floor(0.2*len(ind)), replace=False)
ind_train = list(set(ind) - set(ind_test))

df_train = df_1.iloc[ind_train]
df_test = df_1.iloc[ind_test]

# Création du X_train, y_train, X_test et y_test
X_test = df_test.drop(['Class'], axis='columns')
y_test = df_test['Class']

X_train = df_train.drop(['Class'], axis='columns')
y_train = df_train['Class']


#%%

# Do a random undersampling of dataframe 
def RandomUndersampling(X_train, y_train, target_var_name):

    train = pd.concat([X_train, y_train], axis='columns')

    # We split the data according to the target binary variable
    train_0 = train[train[target_var_name] == 0]
    train_1 = train[train[target_var_name] == 1]

    # The sample size is simply the smallest size
    sample_size = min(len(train_0), len(train_1))

    train_0_resample = train_0.sample(sample_size)
    train_1_resample = train_1.sample(sample_size)

    train_resample = pd.concat([train_0_resample, train_1_resample], axis=0)

    X_train_resample = train_resample.drop([target_var_name], axis='columns')
    y_train_resample = train_resample[target_var_name]

    return X_train_resample, y_train_resample


# Compute the f1-score according to a undersampling (or not) method
def RFClassifier(X_train, X_test, y_train, y_test, nb_splits, method=None):

    t = time.time()
    l_auc = []
    l_recall = []

    probas_fold = np.zeros(len(X_train))
    preds_rf = np.zeros(len(y_test))
    folds = KFold(n_splits=nb_splits, shuffle=True)

    if method == 'TL':
        meth = TomekLinks()
    elif method == 'CC':
        meth = ClusterCentroids()
    elif method == 'NM1':
        meth = NearMiss(version=1, n_neighbors=5)
    elif method == 'NM3':
        meth = NearMiss(version=3, n_neighbors=5)
    elif method == 'SMOTE':
        meth = SMOTE(sampling_strategy='minority', k_neighbors=5)
    else:
        print("Unknown method")
        method = None

    for i, (train_ind, val_ind) in enumerate(folds.split(X_train, y_train)):

        print('fold: {}'.format(i))

        trn_data, val_data = X_train.iloc[train_ind], X_train.iloc[val_ind]
        train_y, y_val = y_train.iloc[train_ind], y_train.iloc[val_ind]

        if method is not None:
            train_resample, y_resample = meth.fit_resample(trn_data, train_y)
        elif method == 'RANDOM':
            train_resample, y_resample = RandomUndersampling(X_train,
                                                             y_train,
                                                             'Class')
        else:
            train_resample, y_resample = trn_data, train_y

        rf = RandomForestClassifier(n_estimators=150, criterion='gini')

        rf.fit(train_resample, y_resample)
    
        probas_fold[val_ind] = rf.predict_proba(val_data)[:, 1]
    
        l_auc.append(roc_auc_score(y_val, probas_fold[val_ind]))

        l_recall.append(
            recall_score(y_val, np.where(probas_fold[val_ind] > 0.5, 1, 0))
            )

        # The average of the probas for all folds
        preds_rf += rf.predict_proba(X_test)[:, 1]/folds.n_splits

        print("Durée fold {}: {}".format(i, time.time() - t))
    
    print('AUC: {}'.format(np.mean(l_auc)))
    print(' Model recall:{}'.format(np.mean(l_recall)))

    preds_train = np.where(probas_fold > 0.5, 1, 0)
    preds = np.where(preds_rf > 0.50, 1, 0)

    print(recall_score(y_test, preds))
    print(classification_report(y_test, preds))

    return preds, preds_train


# Affiche la matrice de confusion
def CF_mat(y_train, preds_train):

    cf_matrix = confusion_matrix(y_train, preds_train)

    group_names = ['TN','FP','FN','TP']

    group_counts = ["{0:0.0f}".format(value)
                    for value in cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value)
                         for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3
              in zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    plt.figure()
    sns.set(font_scale=1.8)
    plt.style.use('seaborn-poster')
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='vlag_r')

    return True



#%%

# 2. Logistic Regression

# 2.a) With Random Undersampling
under_X_train, under_y_train = RandomUndersampling(X_train, y_train, 'Class')

# Test avec Regression logistique
logreg = LogisticRegression()
logreg.fit(under_X_train, under_y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))
# Macro avg: 54%

conf_mat_under = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# 2.b) Model with resampling
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))
# Macro avg de 83%

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
# A lot better

#%%

# 3. Decision Tree Classifier

# 3.a) Without resampling
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
# 0.90

# 3.b) With random undersampling
under_clf = tree.DecisionTreeClassifier()
under_clf.fit(under_X_train, under_y_train)
print(classification_report(y_test, under_clf.predict(X_test)))
# 0.49


#%%

# With Tomek Links
tl_clf = tree.DecisionTreeClassifier()
tl = TomekLinks()
X_train_resamp, y_train_resamp = tl.fit_resample(X_train, y_train)
tl_y_pred = tl_clf.fit(X_train_resamp, y_train_resamp).predict(X_test)
print(classification_report(y_test, tl_y_pred))
# 0.87

#%%

# 4. NAIVE BAYES

# 3.a) Without resampling
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))
# 0.63

# 3.a) With random undersampling
under_gnb = GaussianNB()
under_y_pred = under_gnb.fit(under_X_train, under_y_train).predict(X_test)
print(classification_report(y_test, under_y_pred))
# 0.55

# 3.a) With Tomek Links undersampling
tl_gnb = GaussianNB()
tl_y_pred = tl_gnb.fit(X_train_resamp, y_train_resamp).predict(X_test)
print(classification_report(y_test, tl_y_pred))
# 0.61

#%%

# 5. KNN
knn = KNeighborsClassifier(n_neighbors=5)
y_pred = knn.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))
# With 5 neighbors : f1_score = 0.56

# 5.a) Random Undersampling
knn_under = KNeighborsClassifier(n_neighbors=5)
y_pred_under = knn_under.fit(under_X_train, under_y_train).predict(X_test)
print(classification_report(y_test, y_pred_under))
# 0.39


# 5.b) Tomek Links
knn_under = KNeighborsClassifier(n_neighbors=5)
tl_y_pred = knn_under.fit(X_train_resamp, y_train_resamp).predict(X_test)
print(classification_report(y_test, tl_y_pred))
# 0.51

#%%

# 6. SVC
clf = svm.SVC()
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
# f1 score: 50%


# 6.a) Weighted SVC
wclf = svm.SVC(kernel='linear', class_weight={1: 50})
wclf.fit(X_train, y_train)

y_pred = wclf.predict(X_test)
print(classification_report(y_test, y_pred))


# 6.b) Balanced SVC
model = SVC(gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('SVC balanced : {}'.format(classification_report(y_test, y_pred)))


#%%

# 7. Random Forest
t = time.time()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Random Forest : {}".format(classification_report(y_test, y_pred)))
print(time.time() - t)


#%%

# 8. Weighted Random Forest
t = time.time()
clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Random Forest : {}".format(classification_report(y_test, y_pred)))
print(time.time() - t)


#%%
# The model without resampling
preds, preds_train = RFClassifier(X_train, X_test, y_train, y_test, 3)
CF_mat(y_test, preds)


#%%
# Random Undersampling
preds_rd, preds_train = RFClassifier(X_train, X_test, y_train, y_test, 3, 'RANDOM')
CF_mat(y_test, preds_rd)
print('RANDOM')


#%%
# Near Miss 1 undersampling
preds_nm1, preds_train = RFClassifier(X_train, X_test, y_train, y_test, 3, 'NM1')
CF_mat(y_test, preds_nm1)
print("NM1")


#%%
# Near Miss 3 undersampling
preds_nm3, preds_train = RFClassifier(X_train, X_test, y_train, y_test, 3, 'NM3')
CF_mat(y_test, preds_nm3)
print("NM3")


#%%
# Tomek Links undersampling
preds_tl, preds_train = RFClassifier(X_train, X_test, y_train, y_test, 3, 'TL')
CF_mat(y_test, preds_tl)
print("TL")



#%%

##### WINE QUALITY DATASET #####


chem_2 = ('/Users/baptistehessel/Documents/M2/class_raphael_sourty_DL/'
          'Projet_Imbalanced_Learning/winequalityN.csv')

# Wine Quality
df_2 = pd.read_csv(chem_2)

wine = pd.read_csv(chem_2)

## WINE
wine_quality = wine.quality
wine_vars = wine.drop(columns=['quality'])
# We have some NA values in our dataframe. We decide to remove rows that have NA values 
is_NaN = wine_vars.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = wine_vars[row_has_NaN]
wine_vars = wine_vars.drop(rows_with_NaN.index, axis=0)
wine_vars = wine_vars.reset_index(drop=True)
wine_quality = wine_quality.drop(rows_with_NaN.index, axis=0)
wine_quality = wine_quality.reset_index(drop=True)
wine_vars
# We transform the variable type on dummy variable 
wine_vars = pd.get_dummies(wine_vars, columns=['type'])

wine_vars.info()
# All good
wine_vars.isna().sum()

sns.boxplot(data=wine_vars)
# Could be good to normalize the data


normalized_wine_vars = preprocessing.minmax_scale(wine_vars)

sns.boxplot(data=normalized_wine_vars)
# Better

X_train, X_test, y_train, y_test = train_test_split(normalized_wine_vars,
                                                    wine_quality,
                                                    test_size=0.2,
                                                    random_state=42)

# First the model with Random Forest Classifier
rfcl = RandomForestClassifier(n_estimators=50)
rfcl = rfcl.fit(X_train, y_train)
pred_RF = rfcl.predict(X_test)
acc_RF = accuracy_score(y_test, pred_RF)
print(acc_RF)
# 0.665
# 0.668 with normalization

SV = svm.SVC(C=1,kernel='rbf')
SV.fit(X_train, y_train)
preds_sv = SV.predict(X_test)

print(metrics.accuracy_score(y_test, preds_sv))
# 0.5313

nb = GaussianNB()
nb.fit(X_train, y_train)
preds_nb = nb.predict(X_test)
print(metrics.accuracy_score(y_test, preds_nb))
# 0.2815

lr = LogisticRegression()
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
print(metrics.accuracy_score(y_test, preds_lr))
# 0.5382

dtcl = DecisionTreeClassifier()
dtcl.fit(X_train, y_train)
preds_dtcl = dtcl.predict(X_test)
print(metrics.accuracy_score(y_test, preds_dtcl))
# 0.5583



abcl = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
abcl = abcl.fit(X_train, y_train)
pred_abcl = abcl.predict(X_test)
print(accuracy_score(y_test, pred_abcl))
# 0.4416


gbcl = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
gbcl = gbcl.fit(X_train, y_train)
pred_gbcl = gbcl.predict(X_test)
print(accuracy_score(y_test, pred_gbcl))
# 0.5761



bgcl = BaggingClassifier(n_estimators=50,
                         max_samples= .7,
                         bootstrap=True,
                         oob_score=True)

bgcl = bgcl.fit(X_train, y_train)
pred_bgcl = bgcl.predict(X_test)
print(accuracy_score(y_test, pred_bgcl))
# 0.6427 !



# Random Grid Search for BaggingClassifier
n_estimators = [int(i) for i in range(5, 200, 3)]

max_samples = [.5, .6, .7, .8, .9, 1]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_samples': max_samples}

bgcl_random = RandomizedSearchCV(estimator=bgcl,
                                 param_distributions=random_grid,
                                 n_iter=80,
                                 scoring='neg_mean_absolute_error', 
                                 cv=3,
                                 verbose=2,
                                 random_state=42,
                                 return_train_score=True)

# Fit the random search model
bgcl_random.fit(X_train, y_train)
bgcl_random.best_params_

bgcl = BaggingClassifier(n_estimators=120,
                         max_samples=.7)

bgcl = bgcl.fit(X_train, y_train)
pred_bgcl = bgcl.predict(X_test)
print(accuracy_score(y_test, pred_bgcl))




# Randomized Grid Search for RF Classifier
n_estimators = [int(i) for i in range(50, 200, 10)]
criterion = ['gini', 'entropy']
max_depth = [int(i) for i in range(10)]
max_depth.append(None)
min_samples_split = [int(i) for i in range(1, 5)]
min_samples_leaf = [int(i) for  i in range(4)]
max_features = ["auto", "sqrt", "log2"]
class_weight = ['balanced', 'balanced_subsample', None]

random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features,
               'class_weight': class_weight}

rfcl_random = RandomizedSearchCV(estimator=rfcl,
                                 param_distributions=random_grid,
                                 n_iter=100,
                                 scoring='neg_mean_absolute_error', 
                                 cv=3,
                                 verbose=2,
                                 random_state=42,
                                 return_train_score=True)

# Fit the random search model
rfcl_random.fit(X_train, y_train)
rfcl_random.best_params_

rfcl = RandomForestClassifier(n_estimators=170,
                              min_samples_split=3,
                              min_samples_leaf=2,
                              max_features='auto',
                              max_depth=None,
                              criterion='entropy',
                              class_weight=None)
rfcl = rfcl.fit(X_train, y_train)
pred_RF = rfcl.predict(X_test)
acc_RF = accuracy_score(y_test, pred_RF)
print(acc_RF)
# 0.6566

#%%

##### INSURANCE CLAIMS DATASET #####

chem_3 = ('/Users/baptistehessel/Documents/M2/class_raphael_sourty_DL/'
          'Projet_Imbalanced_Learning/Insurance_Imbalanced/aug_test.csv')

chem_4 = ('/Users/baptistehessel/Documents/M2/class_raphael_sourty_DL/'
          'Projet_Imbalanced_Learning/Insurance_Imbalanced/aug_train.csv')

insurance_train = pd.read_csv(chem_4)

insurance_test = pd.read_csv(chem_3)

insurance_test.columns

## INSURANCE 
insurance_train = insurance_train.drop(columns=['id'])
insurance_class = insurance_train.Response
insurance_vars = insurance_train.drop(columns=['Response'])

# We need to transform categorical variables onto numerical ones in order to
# run different models
# We get dummies for Gender and Vehicle_Damage because they are non ordinal
# variables
insurance_vars = pd.get_dummies(insurance_vars,
                                columns=['Gender', 'Vehicle_Damage'])

# We transform the variable Vehicle_Age (ordinal variable with 3 categories)
# on the variable taking values in {0,1,2}

oe = pre.OrdinalEncoder()
insurance_vars['Vehicle_Age'] = oe.fit_transform(insurance_vars[['Vehicle_Age']])

X_train, X_test, y_train, y_test = train_test_split(insurance_vars,
                                                    insurance_class,
                                                    test_size=0.2,
                                                    random_state=42)


#%%

# Do a random undersampling of dataframe
def RandomUndersampling(X_train, y_train, target_var_name):

    train = pd.concat([X_train, y_train], axis='columns')

    train_0 = train[train[target_var_name] == 0]
    train_1 = train[train[target_var_name] == 1]

    sample_size = min(len(train_0), len(train_1))

    train_0_resample = train_0.sample(sample_size)
    train_1_resample = train_1.sample(sample_size)

    train_resample = pd.concat([train_0_resample, train_1_resample], axis=0)

    X_train_resample = train_resample.drop([target_var_name], axis='columns')
    y_train_resample = train_resample[target_var_name]

    return X_train_resample, y_train_resample


# Compute the f1-score according to a undersampling (or not) method
def RFClassifier(X_train, X_test, y_train, y_test, nb_splits, method=None):

    t = time.time()
    l_auc = []
    l_recall = []

    probas_fold = np.zeros(len(X_train))
    preds_rf = np.zeros(len(y_test))
    folds = KFold(n_splits=nb_splits, shuffle=True)

    if method == 'TL':
        meth = TomekLinks()
    elif method == 'CC':
        meth = ClusterCentroids()
    elif method == 'NM1':
        meth = NearMiss(version=1, n_neighbors=5)
    elif method == 'NM3':
        meth = NearMiss(version=3, n_neighbors=5)
    elif method == 'SMOTE':
        meth = SMOTE(sampling_strategy='minority', k_neighbors=5)
    else:
        print("Unknown method")
        method = None

    for i, (train_ind, val_ind) in enumerate(folds.split(X_train, y_train)):

        print('fold: {}'.format(i))

        trn_data, val_data = X_train.iloc[train_ind], X_train.iloc[val_ind]
        train_y, y_val = y_train.iloc[train_ind], y_train.iloc[val_ind]

        if method is not None:
            train_resample, y_resample = meth.fit_resample(trn_data, train_y)
        elif method == 'RANDOM':
            train_resample, y_resample = RandomUndersampling(X_train,
                                                             y_train,
                                                             'Response')
        else:
            train_resample, y_resample = trn_data, train_y

        rf = RandomForestClassifier(n_estimators=150, criterion='gini')

        rf.fit(train_resample, y_resample)
    
        probas_fold[val_ind] = rf.predict_proba(val_data)[:, 1]
    
        l_auc.append(roc_auc_score(y_val, probas_fold[val_ind]))

        l_recall.append(
            recall_score(y_val, np.where(probas_fold[val_ind] > 0.5, 1, 0))
            )

        # The average of the probas for all folds
        preds_rf += rf.predict_proba(X_test)[:, 1]/folds.n_splits

        print("Durée fold {}: {}".format(i, time.time() - t))
    
    print('AUC: {}'.format(np.mean(l_auc)))
    print(' Model recall:{}'.format(np.mean(l_recall)))

    preds_train = np.where(probas_fold > 0.5, 1, 0)
    preds = np.where(preds_rf > 0.50, 1, 0)

    print(recall_score(y_test, preds))
    print(classification_report(y_test, preds))

    return preds, preds_train


# Affiche la matrice de confusion
def CF_mat(y_train, preds_train):

    cf_matrix = confusion_matrix(y_train, preds_train)

    group_names = ['TN','FP','FN','TP']

    group_counts = ["{0:0.0f}".format(value)
                    for value in cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value)
                         for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3
              in zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    plt.figure()
    sns.set(font_scale=1.8)
    plt.style.use('seaborn-poster')
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='vlag_r')

    return True

#%%

preds, preds_train = RFClassifier(X_train, X_test, y_train, y_test, 3, method=None)
# 0.68
CF_mat(y_test, preds)

preds_tl, preds_train_tl = RFClassifier(X_train, X_test, y_train, y_test, 3, method='TL')
CF_mat(y_test, preds_tl)

preds_nm1, preds_train_nm1 = RFClassifier(X_train, X_test, y_train, y_test, 3, method='NM1')
CF_mat(y_test, preds_nm1)

preds_nm3, preds_train_nm3 = RFClassifier(X_train, X_test, y_train, y_test, 3, method='NM3')
CF_mat(y_test, preds_nm3)


preds_rd, preds_train_rd = RFClassifier(X_train, X_test, y_train, y_test, 3, method='RANDOM')
CF_mat(y_test, preds_rd)

#%%

# 6. SVC
clf = svm.SVC()
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
# f1 score: 50%


# Weighted SVC
wclf = svm.SVC(kernel='linear', class_weight={1: 5})
wclf.fit(X_train, y_train)
y_pred = wclf.predict(X_test)
print(classification_report(y_test, y_pred))

#%%
# Balanced SVC
model = SVC(gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('SVC balanced : {}'.format(classification_report(y_test, y_pred)))

