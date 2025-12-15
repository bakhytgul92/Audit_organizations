#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import shap
import time
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import Adam
from keras import utils as np_utils
from keras import layers
from keras import utils
from keras.metrics import Precision
from keras.metrics import Recall
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Reshape, BatchNormalization
from keras.layers import Flatten
from keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.layers import Layer, Conv1D, Add, GRU
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import RandomOverSampler
from keras.layers import LSTM
from sklearn.utils import shuffle
from scipy.stats import ttest_rel
import itertools


# In[2]:


data = pd.read_csv('audit_data/audit_risk.csv')


# In[3]:


data


# In[4]:


data['Money_Value'] = data['Money_Value'].fillna(data['Money_Value'].mean())


# In[5]:


data = data[data['LOCATION_ID']!='LOHARU']
data = data[data['LOCATION_ID']!='NUH']
data = data[data['LOCATION_ID']!='SAFIDON']


# In[6]:


data['LOCATION_ID'] = data['LOCATION_ID'].astype('int')


# In[7]:


data.describe()


# In[8]:


data.dtypes


# In[9]:


data.Risk.value_counts()


# In[10]:


data['Risk_val'] = np.where(data['Risk']==1, 'Risk', 'Legitimate')


# In[11]:


plt.figure(figsize=[8,8])
data.Risk_val.value_counts().plot(kind='bar', width=0.9, color=['#00F07A','#F5581D'])
plt.xlabel('Class',fontsize=14)
plt.ylabel('Number',fontsize=14)
plt.xticks(rotation='horizontal', fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[12]:


X = data.drop(['Risk', 'Risk_val'], axis=1)


# In[13]:


X


# In[14]:


X.columns


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[16]:


X_scaled = pd.DataFrame(scaler.fit_transform(X))


# In[17]:


X_scaled


# In[18]:


y = data['Risk']


# # Mututal information

# In[19]:


mi_scores = mutual_info_classif(
    X_scaled, y,
    discrete_features=False,
    random_state=42
)


# In[20]:


mi_df = pd.DataFrame({
    "Feature": X_scaled.columns,
    "MI_score": mi_scores
}).sort_values(by="MI_score", ascending=False)


# In[21]:


top_10_mi = mi_df["Feature"].head(10).tolist()


# In[22]:


top_10_mi


# In[23]:


X_scaled_best = X_scaled[top_10_mi]


# In[24]:


X_scaled_best


# In[25]:


y


# In[26]:


selected_features = X.iloc[:, top_10_mi]


# In[27]:


selected_features


# In[28]:


selected_features.drop(['Money_Value', 'PARA_B', 'TOTAL'], axis=1, inplace=True)


# In[29]:


selected_features


# In[30]:


corr_matrix = selected_features.corr()


# In[31]:


plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# In[32]:


ros = RandomOverSampler(random_state=42)


# In[33]:


X_resampled, y_resampled = ros.fit_resample(selected_features, y)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)


# In[35]:


scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(X_train))
x_test  = scaler.transform(X_test) 


# In[36]:


algorithms = ['Naive Bayes','Support vector machine','Decision tree','Random Forest','XGBoost']


# In[37]:


metrics_list = []


# In[38]:


matrix_labels = ['Risk','Legitimate']


# In[39]:


import sklearn.metrics as metrics


# In[40]:


classifiers = [MultinomialNB(), SVC(kernel='linear', probability=True),DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None), 
RandomForestClassifier(n_estimators = 10), xgb.XGBClassifier(random_state=42)]


# In[41]:


x_train.columns = [
    "Audit_Risk", "Inherent_Risk", "Score",
    "Risk_D", "Risk_B", "Score_MV", "Risk_A"
]


# In[42]:


x_train


# In[43]:


x_test = pd.DataFrame(x_test, columns=x_train.columns)


# In[44]:


x_test


# # SHAP plots

# In[45]:


features = ["Audit_Risk", "Inherent_Risk", "Score", "Risk_D", "Risk_B", "Score_MV", "Risk_A"]


# In[46]:


rf = RandomForestClassifier(n_estimators = 10)


# In[47]:


rf.fit(x_train, y_train)


# In[48]:


importances = rf.feature_importances_


# In[49]:


importance_df = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


# In[50]:


colors = [
    "tab:blue", "tab:orange", "tab:green",
    "tab:red", "tab:purple", "tab:brown", "tab:pink"
]


# In[51]:


plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color=colors
)
plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()


# In[52]:


def get_shap_explainer(model, X_background):
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, xgb.XGBClassifier)):
        return shap.TreeExplainer(model)
    else:
        return shap.KernelExplainer(model.predict_proba, X_background)


# In[53]:


for name, model in zip(algorithms, classifiers):
    print(f"\n================ {name} =================")

    model.fit(x_train, y_train)
    background = shap.sample(x_train, 100)
    explainer = get_shap_explainer(model, background)
    shap_values = explainer.shap_values(x_test)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    plt.figure(figsize=(14, 9))
    
    shap.summary_plot(
        shap_vals,
        x_test,
        show=False,
        plot_size=(14, 9),
        max_display=7
    )
        
    plt.suptitle(
        f"SHAP Summary Plot for {name}",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )
    
    plt.xlabel("SHAP value", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# In[54]:


k = 0
cv = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_list = {}
precision_list = {}
recall_list = {}
f1_list = {}

train_time_list = {}
inference_time_list = {}

for i in classifiers:
    start_train = time.perf_counter()
    val_scores = cross_validate(i, x_train, y_train, scoring=['accuracy', 'precision', 'recall', 'f1'], cv=cv)     
    y_pred_train = cross_val_predict(i, x_train, y_train, cv=cv)

    # TRAINING TIME
    end_train = time.perf_counter()
    training_time = end_train - start_train
    train_time_list[algorithms[k]] = training_time

    accuracy_train = val_scores['test_accuracy'].mean()
    accuracy_list[f"{algorithms[k]} accuracy"] = np.round(val_scores['test_accuracy'], 3)
    
    precision_train = val_scores['test_precision'].mean()
    precision_list[f"{algorithms[k]} precision"] = np.round(val_scores['test_precision'], 3)
    
    recall_train = val_scores['test_recall'].mean()
    recall_list[f"{algorithms[k]} recall"] = np.round(val_scores['test_recall'], 3)
    
    f1_train = val_scores['test_f1'].mean()
    f1_list[f"{algorithms[k]} f1"] = np.round(val_scores['test_f1'], 3)
    
    std_accuracy = val_scores['test_accuracy'].std()
    std_precision = val_scores['test_precision'].std()
    std_recall = val_scores['test_recall'].std()
    std_f1 = val_scores['test_f1'].std()

    # INFERENCE TIME
    start_pred = time.perf_counter()
    y_pred_test = i.predict(x_test)
    end_pred = time.perf_counter()

    inference_time_total = end_pred - start_pred
    inference_time_per_sample = inference_time_total / len(x_test)

    inference_time_list[algorithms[k]] = inference_time_per_sample

    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, y_pred_train)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
       
    y_pred_test = i.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test, y_pred_test)
    roc_auc_test = metrics.auc(fpr_test, tpr_test)
    pr_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_test)
    auc_pr = average_precision_score(y_test, y_pred_test)
    
    brier = brier_score_loss(y_test, y_pred_test)

    calib_true, calib_pred = calibration_curve(
        y_test, y_pred_test, n_bins=10, strategy="uniform"
    )
    
    metrics_list.append({'Accuracy': accuracy_test,
                        'Precision': precision_test,
                        'Recall': recall_test,
                        'F1-score': f1_test,
                        'fpr': fpr_test,
                        'tpr': tpr_test,
                        'auc': roc_auc_test,
                        'auc_pr': auc_pr,
                        'pr_curve': pr_curve,
                        'rec_curve': rec_curve,
                        'thresholds_curve': thresholds_curve,
                        'brier': brier,
                        'calib_true': calib_true,
                        'calib_pred': calib_pred})
    
    print("Evaluation metrics of " + algorithms[k]+" algorithm: ")
    
    print("Training scores: ")
    print(f"Accuracy = {accuracy_train:.3f} ± {std_accuracy:.3f}")
    print(f"Precision = {precision_train:.3f} ± {std_precision:.3f}")
    print(f"Recall = {recall_train:.3f} ± {std_recall:.3f}")
    print(f"F1-score = {f1_train:.3f} ± {std_f1:.3f}")

    print("Test scores: ")
    print('Accuracy - ', round(accuracy_test, 3))  
    print('Precision - ', round(precision_test, 3))
    print('Recall - ', round(recall_test, 3))
    print('F1-score - ', round(f1_test, 3))
    print(" ")
    print("Efficiency:")
    print(f"Training time (s): {training_time:.4f}")
    print(f"Inference time per sample (ms): {inference_time_per_sample * 1000:.4f}")
    print(" ")
    k = k + 1


# In[56]:


accuracy_list


# In[57]:


precision_list


# In[58]:


recall_list


# In[59]:


f1_list


# In[60]:


def paired_ttests_each_vs_others(metric_dict):
    models = list(metric_dict.keys())

    print("\nEach model vs all other models (paired t-test)")
    print("-" * 80)

    for model_a in models:
        print(f"\n=== {model_a} ===")
        for model_b in models:
            if model_b == model_a:
                continue

            t_stat, p_value = ttest_rel(metric_dict[model_a], metric_dict[model_b])

            print(f"{model_a:30s} vs {model_b:30s} | "
                  f"t = {t_stat:7.3f} | p = {p_value:.4f}")


# In[61]:


print("Accuracy t-test and p-value scores\n")
paired_ttests_each_vs_others(accuracy_list)


# In[62]:


print("Precision t-test and p-value scores\n")
paired_ttests_each_vs_others(precision_list)


# In[63]:


print("Recall t-test and p-value scores\n")
paired_ttests_each_vs_others(recall_list)


# In[64]:


print("F1-score t-test and p-value scores\n")
paired_ttests_each_vs_others(f1_list)


# In[65]:


x_train.shape


# In[66]:


x_test.shape


# In[67]:


model_dnn = Sequential()
model_dnn.add(Dense(256, input_dim=7, kernel_regularizer=l2(1e-4)))
model_dnn.add(Activation('relu'))
model_dnn.add(Dropout(0.4))
model_dnn.add(Dense(128, kernel_regularizer=l2(1e-4)))
model_dnn.add(Activation('relu'))
model_dnn.add(Dropout(0.2))
model_dnn.add(Dense(1))
model_dnn.add(Activation('sigmoid'))
model_dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall(), tfa.metrics.FBetaScore(num_classes=2,average="micro",threshold=0.9)])
model_dnn.summary()


# In[68]:


start_train = time.perf_counter()


# In[69]:


history_dnn = model_dnn.fit(x_train, y_train, batch_size = 14000, epochs = 100, verbose=1, validation_split=0.125)
loss_dnn, accuracy_dnn, precision_dnn, recall_dnn, f1_score_dnn = model_dnn.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy_dnn))
loss_dnn, accuracy_dnn, precision_dnn, recall_dnn, f1_score_dnn = model_dnn.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy: {:.4f}".format(accuracy_dnn))


# In[70]:


end_train = time.perf_counter()

total_training_time = end_train - start_train
avg_epoch_time = total_training_time / 100

print(f"Total training time (s): {total_training_time:.2f}")
print(f"Average training time per epoch (s): {avg_epoch_time:.2f}")


# In[71]:


#plot for accuracy
plt.rcParams['font.size'] = 16
plt.plot(history_dnn.history['accuracy'], color='#215704')
plt.plot(history_dnn.history['val_accuracy'], color='#A70C13')
plt.title('DNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#plot for loss
plt.plot(history_dnn.history['loss'], color='#215704')
plt.plot(history_dnn.history['val_loss'], color='#A70C13')
plt.title('DNN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[72]:


val_accuracy_list = np.array(history_dnn.history['val_accuracy'])
val_accuracy = round(np.mean(val_accuracy_list), 3)

val_precision_list = np.array(history_dnn.history['val_precision'])
val_precision = round(np.mean(val_precision_list), 3)

val_recall_list = np.array(history_dnn.history['val_recall'])
val_recall = round(np.mean(val_recall_list), 3)

val_f1_list = np.array(history_dnn.history['val_fbeta_score'])
val_f1 = round(np.mean(val_f1_list), 3)


# In[73]:


y_pred_test = model_dnn.predict(x_test)


# In[74]:


y_pred_test = list(map(lambda x: 0 if x<0.5 else 1, y_pred_test))


# In[75]:


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)


# In[76]:


print("Scores:")
print("Accuracy - ", round(accuracy_test, 3))
print("Precision - ", round(precision_test, 3))
print("Recall - ", round(recall_test, 3))
print("F1-score - ", round(f1_score_test, 3))


# In[77]:


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)


# In[78]:


pr_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_test)
auc_pr = average_precision_score(y_test, y_pred_test)


# In[79]:


brier = brier_score_loss(y_test, y_pred_test)

calib_true, calib_pred = calibration_curve(
    y_test, y_pred_test, n_bins=10, strategy="uniform"
)

metrics_list.append({'Accuracy': accuracy_test,
                    'Precision': precision_test,
                    'Recall': recall_test,
                    'F1-score': f1_test,
                    'fpr': fpr_test,
                    'tpr': tpr_test,
                    'auc': roc_auc_test,
                    'auc_pr': auc_pr,
                    'pr_curve': pr_curve,
                    'rec_curve': rec_curve,
                    'thresholds_curve': thresholds_curve,
                    'brier': brier,
                    'calib_true': calib_true,
                    'calib_pred': calib_pred})


# In[80]:


labels = [0,1]


# In[81]:


cm = confusion_matrix(y_test, y_pred_test, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels], yticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for DNN")
plt.show()


# In[82]:


max_features = 10
nb_filter = 250
filter_length = 3
hidden_dims = 250


# In[83]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=5e-5)


# In[84]:


model_cnn = Sequential()
model_cnn.add(Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu', input_shape=(7, 1), kernel_regularizer=l2(1e-4)))
model_cnn.add(BatchNormalization())
model_cnn.add(GlobalMaxPooling1D())

model_cnn.add(Dense(hidden_dims, activation='relu', kernel_regularizer=l2(1e-4)))

model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dense(32, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.00008)

model_cnn.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        tfa.metrics.FBetaScore(num_classes=2,average="micro",threshold=0.9)
    ]
)

model_cnn.summary()


# In[85]:


start_train = time.perf_counter()


# In[86]:


history_cnn = model_cnn.fit(x_train, y_train, batch_size = 2000, epochs = 100, callbacks=[reduce_lr], verbose=1, validation_split=0.125)


# In[87]:


end_train = time.perf_counter()

total_training_time = end_train - start_train
avg_epoch_time = total_training_time / 100

print(f"Total training time (s): {total_training_time:.2f}")
print(f"Average training time per epoch (s): {avg_epoch_time:.2f}")


# In[88]:


#plot for accuracy
plt.rcParams['font.size'] = 16
plt.plot(history_cnn.history['accuracy'], color='#215704')
plt.plot(history_cnn.history['val_accuracy'], color='#A70C13')
plt.title('CNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#plot for loss
plt.plot(history_cnn.history['loss'], color='#215704')
plt.plot(history_cnn.history['val_loss'], color='#A70C13')
plt.title('CNN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[89]:


val_accuracy_list = np.array(history_cnn.history['val_accuracy'])
val_accuracy = round(np.mean(val_accuracy_list), 3)

val_precision_list = np.array(history_cnn.history['val_precision'])
val_precision = round(np.mean(val_precision_list), 3)

val_recall_list = np.array(history_cnn.history['val_recall'])
val_recall = round(np.mean(val_recall_list), 3)

val_f1_list = np.array(history_cnn.history['val_fbeta_score'])
val_f1 = round(np.mean(val_f1_list), 3)


# In[90]:


y_pred_test = model_cnn.predict(x_test)


# In[91]:


y_pred_test = list(map(lambda x: 0 if x<0.5 else 1, y_pred_test))


# In[92]:


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)


# In[93]:


print("Scores training:")
print("Accuracy - ", round(accuracy_test, 3))
print("Precision - ", round(precision_test, 3))
print("Recall - ", round(recall_test, 3))
print("F1-score - ", round(f1_score_test, 3))


# In[94]:


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)


# In[95]:


pr_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_test)
auc_pr = average_precision_score(y_test, y_pred_test)


# In[96]:


brier = brier_score_loss(y_test, y_pred_test)

calib_true, calib_pred = calibration_curve(
    y_test, y_pred_test, n_bins=10, strategy="uniform"
)

metrics_list.append({'Accuracy': accuracy_test,
                    'Precision': precision_test,
                    'Recall': recall_test,
                    'F1-score': f1_test,
                    'fpr': fpr_test,
                    'tpr': tpr_test,
                    'auc': roc_auc_test,
                    'auc_pr': auc_pr,
                    'pr_curve': pr_curve,
                    'rec_curve': rec_curve,
                    'thresholds_curve': thresholds_curve,
                    'brier': brier,
                    'calib_true': calib_true,
                    'calib_pred': calib_pred})


# In[97]:


cm = confusion_matrix(y_test, y_pred_test, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels], yticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for CNN")
plt.show()


# In[98]:


model_lstm = Sequential()
model_lstm.add(LSTM(128, input_shape=(7,1), return_sequences=True, kernel_regularizer=l2(1e-4),))
model_lstm.add(SpatialDropout1D(0.2))
model_lstm.add(LSTM(32, kernel_regularizer=l2(1e-4),))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.0001)
model_lstm.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy', Precision(), Recall(), tfa.metrics.FBetaScore(num_classes=2,average="micro",threshold=0.9)])
model_lstm.summary()


# In[99]:


start_train = time.perf_counter()


# In[100]:


history_lstm = model_lstm.fit(x_train, y_train, batch_size = 14000, epochs = 100, verbose=1, validation_split=0.125)


# In[101]:


end_train = time.perf_counter()

total_training_time = end_train - start_train
avg_epoch_time = total_training_time / 100

print(f"Total training time (s): {total_training_time:.2f}")
print(f"Average training time per epoch (s): {avg_epoch_time:.2f}")


# In[103]:


#plot for accuracy
plt.rcParams['font.size'] = 16
plt.plot(history_lstm.history['accuracy'], color='#215704')
plt.plot(history_lstm.history['val_accuracy'], color='#A70C13')
plt.title('LSTM accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#plot for loss
plt.plot(history_lstm.history['loss'], color='#215704')
plt.plot(history_lstm.history['val_loss'], color='#A70C13')
plt.title('LSTM loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[104]:


val_accuracy_list = np.array(history_lstm.history['val_accuracy'])
val_accuracy = round(np.mean(val_accuracy_list), 3)

val_precision_list = np.array(history_lstm.history['val_precision_1'])
val_precision = round(np.mean(val_precision_list), 3)

val_recall_list = np.array(history_lstm.history['val_recall_1'])
val_recall = round(np.mean(val_recall_list), 3)

val_f1_list = np.array(history_lstm.history['val_fbeta_score'])
val_f1 = round(np.mean(val_f1_list), 3)


# In[105]:


y_pred_test = model_lstm.predict(x_test)


# In[106]:


y_pred_test = list(map(lambda x: 0 if x<0.5 else 1, y_pred_test))


# In[107]:


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)


# In[108]:


print("Scores training:")
print("Accuracy - ", round(accuracy_test, 3))
print("Precision - ", round(precision_test, 3))
print("Recall - ", round(recall_test, 3))
print("F1-score - ", round(f1_score_test, 3))


# In[109]:


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)


# In[110]:


pr_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_test)
auc_pr = average_precision_score(y_test, y_pred_test)


# In[111]:


brier = brier_score_loss(y_test, y_pred_test)

calib_true, calib_pred = calibration_curve(
    y_test, y_pred_test, n_bins=10, strategy="uniform"
)

metrics_list.append({'Accuracy': accuracy_test,
                    'Precision': precision_test,
                    'Recall': recall_test,
                    'F1-score': f1_test,
                    'fpr': fpr_test,
                    'tpr': tpr_test,
                    'auc': roc_auc_test,
                    'auc_pr': auc_pr,
                    'pr_curve': pr_curve,
                    'rec_curve': rec_curve,
                    'thresholds_curve': thresholds_curve,
                    'brier': brier,
                    'calib_true': calib_true,
                    'calib_pred': calib_pred})


# In[112]:


cm = confusion_matrix(y_test, y_pred_test, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels], yticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for LSTM")
plt.show()


# In[113]:


model_cnn_lstm = Sequential()

# CNN layers
model_cnn_lstm.add(Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu', input_shape=(7, 1), kernel_regularizer=l2(1e-4)))
model_cnn_lstm.add(GlobalMaxPooling1D())
model_cnn_lstm.add(Dense(hidden_dims, activation='relu', kernel_regularizer=l2(1e-4)))

# Reshape for LSTM input (batch_size, timesteps, features)
model_cnn_lstm.add(Reshape((1, hidden_dims)))

# LSTM layers
model_cnn_lstm.add(LSTM(128, return_sequences=False))

# Fully connected layers
model_cnn_lstm.add(Dense(64, activation='relu'))
model_cnn_lstm.add(Dense(32, activation='relu'))

# Output layer
model_cnn_lstm.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model_cnn_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, 
                        metrics=['accuracy', Precision(), Recall(), tfa.metrics.FBetaScore(num_classes=2,average="micro",threshold=0.9)])

# Model summary
model_cnn_lstm.summary()


# In[114]:


start_train = time.perf_counter()


# In[115]:


history_cnn_lstm = model_cnn_lstm.fit(x_train, y_train, batch_size = 2000, epochs = 100, verbose=1, validation_split=0.125)


# In[116]:


end_train = time.perf_counter()

total_training_time = end_train - start_train
avg_epoch_time = total_training_time / 100

print(f"Total training time (s): {total_training_time:.2f}")
print(f"Average training time per epoch (s): {avg_epoch_time:.2f}")


# In[117]:


#plot for accuracy
plt.rcParams['font.size'] = 16
plt.plot(history_cnn_lstm.history['accuracy'], color='#215704')
plt.plot(history_cnn_lstm.history['val_accuracy'], color='#A70C13')
plt.title('CNN-LSTM accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#plot for loss
plt.plot(history_cnn_lstm.history['loss'], color='#215704')
plt.plot(history_cnn_lstm.history['val_loss'], color='#A70C13')
plt.title('CNN-LSTM loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[118]:


val_accuracy_list = np.array(history_cnn_lstm.history['val_accuracy'])
val_accuracy = round(np.mean(val_accuracy_list), 3)

val_precision_list = np.array(history_cnn_lstm.history['val_precision_2'])
val_precision = round(np.mean(val_precision_list), 3)

val_recall_list = np.array(history_cnn_lstm.history['val_recall_2'])
val_recall = round(np.mean(val_recall_list), 3)

val_f1_list = np.array(history_cnn_lstm.history['val_fbeta_score'])
val_f1 = round(np.mean(val_f1_list), 3)


# In[119]:


y_pred_test = model_cnn_lstm.predict(x_test)


# In[120]:


y_pred_test = list(map(lambda x: 0 if x<0.5 else 1, y_pred_test))


# In[121]:


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)


# In[122]:


print("Scores training:")
print("Accuracy - ", round(accuracy_test, 3))
print("Precision - ", round(precision_test, 3))
print("Recall - ", round(recall_test, 3))
print("F1-score - ", round(f1_score_test, 3))


# In[123]:


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)


# In[124]:


pr_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_test)
auc_pr = average_precision_score(y_test, y_pred_test)


# In[125]:


brier = brier_score_loss(y_test, y_pred_test)

calib_true, calib_pred = calibration_curve(
    y_test, y_pred_test, n_bins=10, strategy="uniform"
)

metrics_list.append({'Accuracy': accuracy_test,
                    'Precision': precision_test,
                    'Recall': recall_test,
                    'F1-score': f1_test,
                    'fpr': fpr_test,
                    'tpr': tpr_test,
                    'auc': roc_auc_test,
                    'auc_pr': auc_pr,
                    'pr_curve': pr_curve,
                    'rec_curve': rec_curve,
                    'thresholds_curve': thresholds_curve,
                    'brier': brier,
                    'calib_true': calib_true,
                    'calib_pred': calib_pred})


# In[126]:


cm = confusion_matrix(y_test, y_pred_test, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels], yticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for CNN-LSTM")
plt.show()


# In[127]:


model_lstm_gru = Sequential()
model_lstm_gru.add(LSTM(128, input_shape=(7,1), return_sequences=True, kernel_regularizer=l2(1e-4)))
model_lstm_gru.add(SpatialDropout1D(0.25))
model_lstm_gru.add(GRU(64, kernel_regularizer=l2(1e-4)))
model_lstm_gru.add(Dropout(0.2))
model_lstm_gru.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.0001)
model_lstm_gru.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy', Precision(), Recall(), tfa.metrics.FBetaScore(num_classes=2,average="micro",threshold=0.9)])
model_lstm_gru.summary()


# In[128]:


start_train = time.perf_counter()


# In[129]:


history_lstm_gru = model_lstm_gru.fit(x_train, y_train, batch_size = 14000, epochs = 100, verbose=1, validation_split=0.125)


# In[130]:


end_train = time.perf_counter()

total_training_time = end_train - start_train
avg_epoch_time = total_training_time / 100

print(f"Total training time (s): {total_training_time:.2f}")
print(f"Average training time per epoch (s): {avg_epoch_time:.2f}")


# In[131]:


#plot for accuracy
plt.rcParams['font.size'] = 16
plt.plot(history_lstm_gru.history['accuracy'], color='#215704')
plt.plot(history_lstm_gru.history['val_accuracy'], color='#A70C13')
plt.title('LSTM-GRU accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#plot for loss
plt.plot(history_lstm_gru.history['loss'], color='#215704')
plt.plot(history_lstm_gru.history['val_loss'], color='#A70C13')
plt.title('LSTM-GRU loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[133]:


val_accuracy_list = np.array(history_lstm_gru.history['val_accuracy'])
val_accuracy = round(np.mean(val_accuracy_list), 3)

val_precision_list = np.array(history_lstm_gru.history['val_precision_3'])
val_precision = round(np.mean(val_precision_list), 3)

val_recall_list = np.array(history_lstm_gru.history['val_recall_3'])
val_recall = round(np.mean(val_recall_list), 3)

val_f1_list = np.array(history_lstm_gru.history['val_fbeta_score'])
val_f1 = round(np.mean(val_f1_list), 3)


# In[134]:


y_pred_test = model_lstm_gru.predict(x_test)


# In[135]:


y_pred_test = list(map(lambda x: 0 if x<0.5 else 1, y_pred_test))


# In[136]:


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)


# In[137]:


print("Scores training:")
print("Accuracy - ", round(accuracy_test, 3))
print("Precision - ", round(precision_test, 3))
print("Recall - ", round(recall_test, 3))
print("F1-score - ", round(f1_score_test, 3))


# In[138]:


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)


# In[139]:


pr_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_test)
auc_pr = average_precision_score(y_test, y_pred_test)


# In[140]:


brier = brier_score_loss(y_test, y_pred_test)

calib_true, calib_pred = calibration_curve(
    y_test, y_pred_test, n_bins=10, strategy="uniform"
)

metrics_list.append({'Accuracy': accuracy_test,
                    'Precision': precision_test,
                    'Recall': recall_test,
                    'F1-score': f1_test,
                    'fpr': fpr_test,
                    'tpr': tpr_test,
                    'auc': roc_auc_test,
                    'auc_pr': auc_pr,
                    'pr_curve': pr_curve,
                    'rec_curve': rec_curve,
                    'thresholds_curve': thresholds_curve,
                    'brier': brier,
                    'calib_true': calib_true,
                    'calib_pred': calib_pred})


# In[141]:


cm = confusion_matrix(y_test, y_pred_test, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels], yticklabels=["Legitimate" if lbl == 0 else "Risk" for lbl in labels])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for LSTM-GRU")
plt.show()


# In[142]:


len(metrics_list)


# In[143]:


ML_DNN_models = ['Naive Bayes','Support vector machine','Decision tree','Random Forest','XGBoost','DNN','CNN','LSTM','CNN-LSTM','LSTM-GRU']


# In[144]:


metrics_list


# In[145]:


data_bar = pd.DataFrame(metrics_list, index=ML_DNN_models)


# In[146]:


data_bar


# In[147]:


data_fpr_tpr = pd.DataFrame(metrics_list, index=ML_DNN_models)
data_fpr_tpr.drop(['Accuracy','Precision','Recall','F1-score'], axis=1, inplace=True)


# In[148]:


data_bar.drop(['fpr','tpr','auc'], axis=1, inplace=True)


# In[149]:


tpr_list = data_fpr_tpr['tpr'].to_list()
fpr_list = data_fpr_tpr['fpr'].to_list()


# In[150]:


tpr_list


# In[151]:


auc_roc_list = data_fpr_tpr['auc'].values


# In[152]:


auc_roc_list


# In[153]:


auc_pr_list = data_fpr_tpr['auc_pr'].values


# In[154]:


precision_list = data_bar['Precision']
recall_list = data_bar['Recall']


# In[155]:


data_fpr_tpr


# In[156]:


pr_curve_list = data_fpr_tpr['pr_curve']
rec_curve_list = data_fpr_tpr['rec_curve']
brier_list = data_fpr_tpr['brier']
calib_true_list = data_fpr_tpr['calib_true']
calib_pred_list = data_fpr_tpr['calib_pred']


# In[157]:


clrs2 = ['green', 'orange', 'red', 'blue', 'cyan', '#E02BE0', '#57BFFA', '#74bdcb', '#ffaebc', '#60a3d9']


# In[158]:


bar_plot_data = pd.DataFrame({
    "Accuracy": data_bar["Accuracy"].values,
    "Precision": data_bar["Precision"].values,
    "Recall": data_bar["Recall"].values,
    "F1-score": data_bar["F1-score"].values
    },
    index=['Naive Bayes','Support vector machine','Decision tree','Random Forest','XGBoost','DNN','CNN','LSTM','CNN-LSTM','LSTM-GRU']
)


# In[159]:


fig, axes = plt.subplots(4, 1, figsize=(10,30))
fig.suptitle('Plots', fontsize = 16)
axes[0].set_title('Classification metrics', fontsize = 16)
clrs = ['#3F1FE0',  '#F43BFA', '#F08712', '#12FA72']
data_bar.plot(kind="bar", ax=axes[0], color=clrs, width=0.8)
axes[0].set_xlabel('Classifiers', fontsize = 14)
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_ylabel('Values', fontsize = 14)
axes[0].legend().remove()

axes[1].set_title('AUC ROC curves', fontsize = 14)
axes[1].plot([0, 1], [0, 1],'r--')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].set_ylabel('True Positive Rate', fontsize = 16)
axes[1].set_xlabel('False Positive Rate', fontsize = 16)
for i in range(len(auc_roc_list)):
    axes[1].plot(fpr_list[i], tpr_list[i], clrs2[i], label = 'AUC_ROC for ' + ML_DNN_models[i] + ' = %0.2f' % auc_roc_list[i])

axes[1].legend(loc = 'lower right', fancybox=True, fontsize=10, shadow = True)

axes[2].set_title('Precision-Recall (AUC-PR) Curves', fontsize=14)

axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.5, 1.0])

axes[2].set_xlabel('Recall', fontsize=16)
axes[2].set_ylabel('Precision', fontsize=16)

for i in range(len(auc_pr_list)):
    axes[2].plot(
        rec_curve_list[i],
        pr_curve_list[i],
        linewidth=2,           
        label=f"AUC_PR for {ML_DNN_models[i]} = {auc_pr_list[i]:.2f}"
    )

axes[2].legend(loc='lower left', fancybox=True, fontsize=10, shadow=True)

axes[3].set_title('Calibration Curves', fontsize=14)
axes[3].set_xlim([0.0, 1.0])
axes[3].set_ylim([0.0, 1.0])
axes[3].set_xlabel('Predicted Probability', fontsize=16)
axes[3].set_ylabel('Observed Frequency', fontsize=16)
axes[3].plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

for i in range(len(brier_list)):
    axes[3].plot(
        calib_pred_list[i],
        calib_true_list[i],
        marker='o',
        label=f"Brier for {ML_DNN_models[i]} = {brier_list[i]:.2f}"
    )

axes[3].legend(loc='lower right', fancybox=True, fontsize=10, shadow=True)

fig.tight_layout()
plt.show()


# In[ ]:




