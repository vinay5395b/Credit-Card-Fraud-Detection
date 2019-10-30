import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.stats as stats
import seaborn as sns

df = pd.read_csv('creditcard.csv')

print(df.shape[0], df.shape[1])

df.sample(5)
df.info()

df.loc[:, ['Time','Amount']].describe()

sns.distplot(df.Time)

plt.title('Distribution of Monetary Value Feature')
sns.distplot(df.Amount)

#fraud vs normal transactions
counts = df.Class.value_counts()
normal = counts[0]
fraudulent = counts[1]

normal_perc = (normal/(normal+fraudulent))*100
fraudulent_perc = (fraudulent/(normal+fraudulent))*100
print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(normal,normal_perc,fraudulent,fraudulent_perc))

plt.xlabel('Class-> 0: Non-Fraudulent 1:Fraudulent')
plt.ylabel('Count')
sns.barplot(x = counts.index, y = counts)


corr = df.corr()
corr
heat = sns.heatmap(data=corr)
plt.show(heat)

#skewness
skew = df.skew()
skew

#Scale Amount and Time

from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()

#scale time
scaled_time = scaler1.fit_transform(df[['Time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time = pd.Series(flat_list1)

#scale amount
scaled_amount = scaler2.fit_transform(df[['Amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list2)

df = pd.concat([df, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)

df.head()

df.drop(['Amount','Time'], axis=1, inplace=True)


#Splitting

mask = np.random.rand(len(df)) < 0.9
train = df[mask]
test = df[~mask]

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


no_of_frauds = train.Class.value_counts()[1]
no_of_frauds

non_fraud = train.Class.value_counts()[0]
non_fraud

non_fraud = train[train['Class']==0]
fraud = train[train['Class']==1]

selected = non_fraud.sample(no_of_frauds)
selected.head()

selected.reset_index(drop=True, inplace=True)
fraud.reset_index(drop=True, inplace=True)

sub_sample = pd.concat([selected,fraud])
sub_sample.shape

#shuffle the data
sub_sample = sub_sample.sample(frac=1).reset_index(drop=True)

new_counts = sub_sample.Class.value_counts()
plt.xlabel('Class: 0-Legitimate 1-Fraudulent')
sns.barplot(x=new_counts.index, y=new_counts)

corr = sub_sample.corr()
corr = corr[['Class']]
corr

#negative corr smaller than -0.5
corr[corr.Class<-0.5]

#positive corr
corr[corr.Class>0.5]


#visualizing the features with high negative corr

f, axes = plt.subplots(nrows=2, ncols=4, figsize=(26,16))
f.suptitle('features with high negative corr')
sns.boxplot(x='Class',y='V3', data=sub_sample, ax=axes[0,0])
sns.boxplot(x='Class',y='V9', data=sub_sample, ax=axes[0,1])
sns.boxplot(x='Class',y='V10', data=sub_sample, ax=axes[0,2])
sns.boxplot(x='Class',y='V12', data=sub_sample, ax=axes[0,3])
sns.boxplot(x='Class',y='V14', data=sub_sample, ax=axes[1,0])
sns.boxplot(x='Class',y='V16', data=sub_sample, ax=axes[1,1])
sns.boxplot(x='Class',y='V17', data=sub_sample, ax=axes[1,2])
f.delaxes(axes[1,3])


#visualizing the features w high positive correlation
f, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))

f.suptitle('Features With High Positive Correlation', size=20)
sns.boxplot(x="Class", y="V4", data=sub_sample, ax=axes[0])
sns.boxplot(x="Class", y="V11", data=sub_sample, ax=axes[1])


#Only removing extreme outliers
Q1 = sub_sample.quantile(0.25)
Q1
Q3 = sub_sample.quantile(0.75)
IQR = Q3 - Q1

df2 = sub_sample[~((sub_sample < (Q1 - 2.5 * IQR)) |(sub_sample > (Q3 + 2.5 * IQR))).any(axis=1)]


from sklearn.manifold import TSNE
X = df2.drop('Class',axis=1)
y = df2['Class']


# t-SNE
X_reduced_tsne = TSNE(n_components = 2, random_state=42).fit_transform(X.values)

# t-SNE scatterplot
import matplotlib.patches as mpatches

f, ax = plt.subplots(figsize=(24,16))


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax.set_title('t-SNE', fontsize=14)

ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])

## CLASSIFCATION

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_validation = X_test.values
y_train = y_train.values
y_validation = y_test.values


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier()))
models.append(('RF', RandomForestClassifier()))

# testing models

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = '%s: %F (%F)' %(name, cv_results.mean(), cv_results.std())
    print(msg)

#compare algorithms
    
#Compare Algorithms

fig = plt.figure(figsize=(12,10))
plt.title('Comparison of Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('ROC-AUC Score')
plt.boxplot(results)
ax = fig.add_subplot(111)
ax.set_xticklabels(names)
plt.show()



#from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
#
#model_LR = LogisticRegression()
#model_LR.fit(X_train, y_train)
#
#predictions = model_LR.predict(X_validation)
#score = model_LR.score(X_validation, y_validation)
#score
#
#y_score_lr = model_LR.predict_proba(X_validation)[:,-1]
#
#avg_prec = average_precision_score(y_validation, y_score_lr)
#format(avg_prec)
#
#precision, recall, _ = precision_recall_curve(y_validation, y_score_lr)
#
#plt.step(recall, precision, color='b', alpha=0.2,
#         where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2,
#                 color='b')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#          avg_prec))
#
#
#fpr_rf, tpr_rf, _ = roc_curve(y_validation, y_score_lr)
#roc_auc_rf = auc(fpr_rf, tpr_rf)
#plt.figure(figsize=(8,8))
#plt.xlim([-0.01, 1.00])
#plt.ylim([-0.01, 1.01])
#plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))
#
#
#plt.xlabel('False Positive Rate', fontsize=16)
#plt.ylabel('True Positive Rate', fontsize=16)
#plt.title('ROC curve', fontsize=16)
#plt.legend(loc='lower right', fontsize=13)
#plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
#plt.axes().set_aspect('equal')
#plt.show()
#
#
#predictions
#y_validation
#
#print(classification_report(y_validation, predictions))



