#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:10:19 2024

@author: fehmi
"""
"""
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome

number_of_pregnancy Number of times pregnant.
glucose Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
blood_pressure Diastolic blood pressure (mm Hg).
skin_thickness Triceps skinfold thickness (mm).
insulin 2-Hour serum insulin (mu U/ml).
bmi  Body mass index (weight in kg/(height in m)^2).
diabetes_pedigree Diabetes pedigree function.
age  Age (years).
outcome  Class variable (0 or 1).
"""

file_name='veri-seti.txt';
column_names = ['number_of_pregnancy', 'glucose', 'blood_pressure','skin_thickness','insulin','bmi','diabetes_pedigree','age','outcome']


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mstats



def fn_calc_mean(dfx,col_name):
    df_tmp = dfx[dfx[col_name]>0]
    return df_tmp[col_name].mean()

"""
def fn_calc_mean_without_outlier(dfx,col_name):
    df_tmp = dfx[dfx[col_name]>0]

    # IQR hesaplaması
    Q1 = df_tmp[col_name].quantile(0.25)
    Q3 = df_tmp[col_name].quantile(0.75)
    IQR = Q3 - Q1
    # Aykırı değer olmayan ve col_name > 0 olan veriler
    filtered_df = df_tmp[ ~((df_tmp[col_name] < (Q1 - 1.5 * IQR)) | (df_tmp[col_name] > (Q3 + 1.5 * IQR)))]

    return df_tmp[col_name].mean()
"""


df = pd.read_csv(file_name,sep='\t',names=column_names);

df2 = df;

# Histogram
df2.hist(bins=50, figsize=(20, 15))
plt.suptitle("Original Data Histogram")
plt.show()

df2.plot(kind='density', subplots=True, layout=(3,3), figsize=(20, 15), sharex=False,title="Original Data Density")
plt.show()

# df2['pregnancy_sqrt'] = np.sqrt(df2['number_of_pregnancy']);

df_normal = df2[df2['outcome']==0]
df_diabetes = df2[df2['outcome']==1]

naValues = ["glucose", "blood_pressure", "skin_thickness", "insulin", "age","bmi","diabetes_pedigree" ]
for i in naValues:
    # Normal olanlarda 0'ları dolduralım
    ortalama = fn_calc_mean(df_normal, i)
    ortalama2 = fn_calc_mean(df_diabetes, i)
    if (i=="bmi") | (i=="diabetes_pedigree"):
        df2.loc[(df2[i] == 0) & (df2['outcome']==0), i] = ortalama
        df2.loc[(df2[i] == 0) & (df2['outcome']==1), i] = ortalama2
    else:
        df2.loc[(df2[i] == 0) & (df2['outcome']==0), i] = int(ortalama)
        df2.loc[(df2[i] == 0) & (df2['outcome']==1), i] = int(ortalama2)
    
    df2[i] = mstats.winsorize(df2[i], limits=[0.05, 0.05])
    

from sklearn import preprocessing

df_n = pd.DataFrame(preprocessing.normalize(df2),columns=["number_of_pregnancy","glucose", "blood_pressure", "skin_thickness", "insulin", "age","bmi","diabetes_pedigree","outcome"])


fig = plt.figure(figsize =(10, 5))

# 'KolonAdı' isimli kolona göre boxplot çizelim
df_n.boxplot(column=["glucose", "blood_pressure", "skin_thickness", "insulin", "age","bmi","diabetes_pedigree" ])

# Grafik başlığı ekleyelim
plt.title('preprocessing.normalize BoxPlot')

# Y ekseni etiketi
plt.ylabel('Değerler')

# Görseli gösterelim
plt.show()


# Örneğin, alt ve üst %5 değerleri sınırlandıralım


# Histogram
df2.hist(bins=50, figsize=(20, 15))
plt.show()

df2.plot(kind='density', subplots=True, layout=(3,3), figsize=(20, 15), sharex=False)
plt.show()


"""
scaler = StandardScaler()

# DataFrame'i ölçeklendirme
df_standardized = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)

# Standartlaştırılmış DataFrame'i yazdırma
print(df_standardized)

    
print(df2['diabetes_pedigree'].describe())
"""
# Min-Max Scaler tanımlayalım
scaler = StandardScaler()

# DataFrame'i ölçeklendirme
df_normalized = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)

fig = plt.figure(figsize =(10, 5))

# 'KolonAdı' isimli kolona göre boxplot çizelim
df_normalized.boxplot(column=["glucose", "blood_pressure", "skin_thickness", "insulin", "age","bmi","diabetes_pedigree" ])

# Grafik başlığı ekleyelim
plt.title('Winsorized BoxPlot')

# Y ekseni etiketi
plt.ylabel('Değerler')

# Görseli gösterelim
plt.show()



y = df2.outcome
X = df_normalized.drop('outcome',axis=1)



# unique_classes = np.unique(y)
# n_features = X.shape[1]
# n_components = min(n_features, len(unique_classes) - 1)


##### PCA  &  LDA

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

print(pca.components_)

# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)

fig1, ax1 = plt.subplots()
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")

plt.show()



lda = LinearDiscriminantAnalysis(n_components=1)
X_r2 = lda.fit(X, y).transform(X)

print(lda.scalings_)

col_names = X.columns;
print(X.columns);

lda_components = lda.scalings_.flatten()
for i,component in enumerate(lda_components):
    col_name = col_names[i]
    print(f"{col_name} : {component}")


#### Train Test split yapalım

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(principalComponents, y, test_size=0.3, random_state=42, stratify=y)


from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score,classification_report
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import seaborn as sns


########### LINEAR REGRESSION

from sklearn.linear_model import LinearRegression, LogisticRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma (sürekli değerler olarak)
y_pred_continuous = model.predict(X_test)

# Tahmin edilen sürekli değerleri ikili sınıflara dönüştürme
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred_continuous]


# Doğruluk (Accuracy)
accuracy = accuracy_score(y_test, y_pred_binary)
print("Linear Regression Accuracy:", accuracy)

# Hassasiyet (Sensitivity) or Recall
sensitivity = recall_score(y_test, y_pred_binary, pos_label=1)  # '1' pozitif sınıf olarak kabul edilir
print("Linear Regression Sensitivity:", sensitivity)

# Kesinlik (Precision)
precision = precision_score(y_test, y_pred_binary, pos_label=1)
print("Linear Regression Precision:", precision)

# F1-Skor
f1 = f1_score(y_test, y_pred_binary, pos_label=1)
print("Linear Regression F1 Score:", f1)


# Modelin test seti üzerindeki tahminlerini al
y_scores = model.predict(X_test)

# Gerçek pozitif oranı (TPR) ve yanlış pozitif oranı (FPR) hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# AUC değerini hesapla
roc_auc = auc(fpr, tpr)

# ROC Eğrisi çizdir
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Linear Regression (ROC)')
plt.legend(loc="lower right")
plt.show()



# Confusion Matrix hesaplama
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Lineer Regresyon Confusion Matrix:\n", conf_matrix)

# Confusion matrix çiz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('Lineer Regresyon Confusion Matrix')
plt.show()

########################################


########### PCA LINEAR REGRESSION

model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train_pca)

# Test seti üzerinde tahmin yapma (sürekli değerler olarak)
y_pred_continuous_pca = model_pca.predict(X_test_pca)

# Tahmin edilen sürekli değerleri ikili sınıflara dönüştürme
y_pred_binary_pca = [1 if x >= 0.5 else 0 for x in y_pred_continuous_pca]


# Doğruluk (Accuracy)
accuracy_pca = accuracy_score(y_test_pca, y_pred_binary_pca)
print("PCA Accuracy:", accuracy_pca)

# Hassasiyet (Sensitivity) or Recall
sensitivity_pca = recall_score(y_test_pca, y_pred_binary_pca, pos_label=1)  # '1' pozitif sınıf olarak kabul edilir
print("PCA Sensitivity:", sensitivity_pca)

# Kesinlik (Precision)
precision_pca = precision_score(y_test_pca, y_pred_binary_pca, pos_label=1)
print("PCA Precision:", precision_pca)

# F1-Skor
f1_pca = f1_score(y_test_pca, y_pred_binary_pca, pos_label=1)
print("PCA F1 Score:", f1_pca)


# Modelin test seti üzerindeki tahminlerini al
y_scores_pca = model_pca.predict(X_test_pca)

# Gerçek pozitif oranı (TPR) ve yanlış pozitif oranı (FPR) hesapla
fpr, tpr, thresholds = roc_curve(y_test_pca, y_scores_pca)

# AUC değerini hesapla
roc_auc = auc(fpr, tpr)

# ROC Eğrisi çizdir
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'PCA ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA Linear Regression (ROC)')
plt.legend(loc="lower right")
plt.show()



# Confusion Matrix hesaplama
conf_matrix_pca = confusion_matrix(y_test_pca, y_pred_binary_pca)
print("PCA Lineer Regresyon Confusion Matrix:\n", conf_matrix_pca)

# Confusion matrix çiz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_pca, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('PCA Lineer Regresyon Confusion Matrix')
plt.show()


########################################



####### LOGISTIC REGRESSION

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Sınıf 1 için olasılıklar

# Başarı metrikleri
print("Lojistik Regresyon Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Lojistik Regresyon Classification Report:\n", classification_report(y_test, y_pred))

# ROC Eğrisi ve AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ROC Eğrisi çizimi
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Lojistik Regresyon (ROC)')
plt.legend(loc="lower right")
plt.show()


# Confusion matrix çiz
conf_matrix_lojistik = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lojistik, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('Lojistik Regresyon Confusion Matrix')
plt.show()

###########################################




####### PCA LOGISTIC REGRESSION

model_pca  = LogisticRegression()
model_pca.fit(X_train_pca, y_train_pca)

y_pred_pca = model_pca.predict(X_test_pca)
y_prob_pca = model_pca.predict_proba(X_test_pca)[:, 1]  # Sınıf 1 için olasılıklar

# Başarı metrikleri
print("PCA Lojistik Regresyon Confusion Matrix:\n", confusion_matrix(y_test_pca, y_pred_pca))
print("PCA Lojistik Regresyon Classification Report:\n", classification_report(y_test_pca, y_pred_pca))

# ROC Eğrisi ve AUC
fpr, tpr, _ = roc_curve(y_test_pca, y_prob_pca)
roc_auc = auc(fpr, tpr)

# ROC Eğrisi çizimi
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA Lojistik Regresyon (ROC)')
plt.legend(loc="lower right")
plt.show()


# Confusion matrix çiz
conf_matrix_lojistik_pca = confusion_matrix(y_test_pca, y_pred_pca)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lojistik_pca, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('PCA Lojistik Regresyon Confusion Matrix')
plt.show()

###########################################



########## DECISION TREE ###########

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=4, 
    min_samples_split=10, 
    min_samples_leaf=4,
    max_leaf_nodes=10
    )
clf = clf.fit(X_train, y_train)

plt.figure(figsize=(100,50))  # Figür boyutunu büyüt
tree.plot_tree(clf)

########## Confusion Matrix ########

# Test seti üzerinde tahmin yapma
y_pred = clf.predict(X_test)


# Doğruluk (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Karar Ağacı Accuracy:", accuracy)

# Hassasiyet (Sensitivity) or Recall
sensitivity = recall_score(y_test, y_pred, pos_label=1)  # '1' pozitif sınıf olarak kabul edilir
print("Karar Ağacı Sensitivity:", sensitivity)

# Kesinlik (Precision)
precision = precision_score(y_test, y_pred, pos_label=1)
print("Karar Ağacı Precision:", precision)

# F1-Skor
f1 = f1_score(y_test, y_pred, pos_label=1)
print("Karar Ağacı F1 Score:", f1)

# Özgüllük (Specificity)
# tn, fp, fn, tp = clf.ravel()
# specificity = tn / (tn + fp)
# print("Specificity:", specificity)


# Confusion Matrix hesaplama
conf_matrix = confusion_matrix(y_test, y_pred)
print("Karar Ağacı Confusion Matrix:\n", conf_matrix)

# Confusion matrix çiz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('Karar Ağacı Confusion Matrix')
plt.show()

######### ROC Eğrisi


# Tahmin olasılıklarını al (pozitif sınıf için)
y_scores = clf.predict_proba(X_test)[:, 1]

# ROC eğrisi için TPR, FPR değerlerini hesaplama
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# AUC değerini hesaplama
roc_auc = auc(fpr, tpr)
print("Karar Ağacı ROC AUC:", roc_auc)

# ROC Eğrisi çizimi
plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Example Model').plot()
plt.title('Karar Ağacı (ROC)')
plt.show()

########################################




########## PCA DECISION TREE ###########

clf_pca = tree.DecisionTreeClassifier(max_depth=4, 
    min_samples_split=10, 
    min_samples_leaf=4,
    max_leaf_nodes=10
    )
clf_pca = clf_pca.fit(X_train_pca, y_train_pca)

plt.figure(figsize=(100,50))  # Figür boyutunu büyüt
tree.plot_tree(clf_pca)

########## Confusion Matrix ########

# Test seti üzerinde tahmin yapma
y_pred = clf_pca.predict(X_test_pca)


# Doğruluk (Accuracy)
accuracy = accuracy_score(y_test_pca, y_pred_pca)
print("PCA Karar Ağacı Accuracy:", accuracy)

# Hassasiyet (Sensitivity) or Recall
sensitivity = recall_score(y_test_pca, y_pred_pca, pos_label=1)  # '1' pozitif sınıf olarak kabul edilir
print("PCA Karar Ağacı Sensitivity:", sensitivity)

# Kesinlik (Precision)
precision = precision_score(y_test_pca, y_pred_pca, pos_label=1)
print("PCA Karar Ağacı Precision:", precision)

# F1-Skor
f1 = f1_score(y_test_pca, y_pred_pca, pos_label=1)
print("PCA Karar Ağacı F1 Score:", f1)

# Özgüllük (Specificity)
# tn, fp, fn, tp = clf.ravel()
# specificity = tn / (tn + fp)
# print("Specificity:", specificity)


# Confusion Matrix hesaplama
conf_matrix_pca = confusion_matrix(y_test_pca, y_pred_pca)
print("PCA Karar Ağacı Confusion Matrix:\n", conf_matrix_pca)

# Confusion matrix çiz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_pca, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('PCA Karar Ağacı Confusion Matrix')
plt.show()

######### ROC Eğrisi


# Tahmin olasılıklarını al (pozitif sınıf için)
y_scores = clf_pca.predict_proba(X_test_pca)[:, 1]

# ROC eğrisi için TPR, FPR değerlerini hesaplama
fpr, tpr, thresholds = roc_curve(y_test_pca, y_scores_pca)

# AUC değerini hesaplama
roc_auc = auc(fpr, tpr)
print("PCA Karar Ağacı ROC AUC:", roc_auc)

# ROC Eğrisi çizimi
plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='PCA Karar Ağacı ROC AUC').plot()
plt.title('PCA Karar Ağacı (ROC)')
plt.show()

########################################


#### NAIVE BAYES ##############

from sklearn.naive_bayes import BernoulliNB

# Bernoulli Naive Bayes modelini oluşturma ve eğitme
model = BernoulliNB()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)




# Doğruluk (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)

# Hassasiyet (Sensitivity) or Recall
sensitivity = recall_score(y_test, y_pred, pos_label=1)  # '1' pozitif sınıf olarak kabul edilir
print("Naive Bayes Sensitivity:", sensitivity)

# Kesinlik (Precision)
precision = precision_score(y_test, y_pred, pos_label=1)
print("Naive Bayes Precision:", precision)

# F1-Skor
f1 = f1_score(y_test, y_pred, pos_label=1)
print("Naive Bayes F1 Score:", f1)

# Özgüllük (Specificity)
# tn, fp, fn, tp = clf.ravel()
# specificity = tn / (tn + fp)
# print("Specificity:", specificity)


# Confusion Matrix hesaplama
conf_matrix = confusion_matrix(y_test, y_pred)
print("Naive Bayes Confusion Matrix:\n", conf_matrix)

# Confusion matrix çiz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('Naive Bayes Confusion Matrix')
plt.show()



######### ROC Eğrisi
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# Tahmin olasılıklarını al (pozitif sınıf için)
y_scores = model.predict_proba(X_test)[:, 1]

# ROC eğrisi için TPR, FPR değerlerini hesaplama
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# AUC değerini hesaplama
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# ROC Eğrisi çizimi
plt.figure(figsize=(8, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Naive Bayes BernoulliNB').plot()
plt.title('Naive Bayes (ROC)')
plt.show()

########################################




#### PCA NAIVE BAYES ##############


# Bernoulli Naive Bayes modelini oluşturma ve eğitme
model_pca = BernoulliNB()
model_pca.fit(X_train_pca, y_train_pca)

# Test seti üzerinde tahmin yapma
y_pred = model_pca.predict(X_test_pca)




# Doğruluk (Accuracy)
accuracy = accuracy_score(y_test_pca, y_pred_pca)
print("PCA Naive Bayes Accuracy:", accuracy)

# Hassasiyet (Sensitivity) or Recall
sensitivity = recall_score(y_test_pca, y_pred_pca, pos_label=1)  # '1' pozitif sınıf olarak kabul edilir
print("PCA Naive Bayes Sensitivity:", sensitivity)

# Kesinlik (Precision)
precision = precision_score(y_test_pca, y_pred_pca, pos_label=1)
print("PCA Naive Bayes Precision:", precision)

# F1-Skor
f1 = f1_score(y_test_pca, y_pred_pca, pos_label=1)
print("PCA Naive Bayes F1 Score:", f1)

# Özgüllük (Specificity)
# tn, fp, fn, tp = clf.ravel()
# specificity = tn / (tn + fp)
# print("Specificity:", specificity)


# Confusion Matrix hesaplama
conf_matrix_pca = confusion_matrix(y_test_pca, y_pred_pca)
print("PCA Naive Bayes Confusion Matrix:\n", conf_matrix_pca)

# Confusion matrix çiz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_pca, annot=True, fmt="d", cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('PCA Naive Bayes Confusion Matrix')
plt.show()



######### ROC Eğrisi

# Tahmin olasılıklarını al (pozitif sınıf için)
y_scores_pca = model_pca.predict_proba(X_test_pca)[:, 1]

# ROC eğrisi için TPR, FPR değerlerini hesaplama
fpr, tpr, thresholds = roc_curve(y_test_pca, y_scores_pca)

# AUC değerini hesaplama
roc_auc = auc(fpr, tpr)
print("PCA Naive Bayes AUC:", roc_auc)

# ROC Eğrisi çizimi
plt.figure(figsize=(8, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='PCA Naive Bayes BernoulliNB').plot()
plt.title('PCA Naive Bayes (ROC)')
plt.show()

########################################
