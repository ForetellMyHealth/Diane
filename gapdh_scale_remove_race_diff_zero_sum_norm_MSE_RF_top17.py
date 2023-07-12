#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model, Sequential 
from tensorflow.keras import backend as K
early_stopping = EarlyStopping(patience=10)

# Helper libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn import metrics
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import joblib

# %%
from tensorflow.keras import layers
import kerastuner as kt
from kerastuner import tuners

# %%
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(2021)

# %%
def check_correct(predict, y):
    result = {}
    result['True-Positive'] = 0
    result['True-Negative'] = 0
    result['False-Negative'] = 0
    result['False-Positive'] = 0

    for i in range(len(predict)) :
        if predict[i] == y[i] :
            if y[i] == 0 :
                result['True-Negative'] += 1
            else :
                result['True-Positive'] += 1
        else :
            if y[i] == 0 :
                result['False-Positive'] += 1
            else :
                result['False-Negative'] += 1

    for result_k, result_v in result.items():
        print(result_k +" : "+ str(result_v))
    
    acc=(result['True-Positive']+result['True-Negative'])/len(y)
    sensitivity=result['True-Positive']/(result['True-Positive']+result['False-Negative'])
    specificity=result['True-Negative']/(result['True-Negative']+result['False-Positive'])

    print("Accuracy : ", acc)
    print("Sensitivity :", sensitivity)
    print("Specificity :", specificity)
    
    return acc, sensitivity, specificity

# %%
def custom_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float')
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    sen=tp/(tp+fn+K.epsilon())
    spe=tn/(tn+fp+K.epsilon())
    #p = tp / (tp + fp + K.epsilon())
    #r = tp / (tp + fn + K.epsilon())
    bal=(sen+spe)/(2+K.epsilon())
    bal = tf.where(tf.math.is_nan(bal), tf.zeros_like(bal), bal)
    #f1 = 2*p*r / (p+r+K.epsilon())
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    print(tf.size(tp),tf.size(tn))
    return 1 - K.mean(bal)

# %%
def model_performance(x, model, y, sample):
    hypo=model.predict(x)
    pred = np.where(hypo > 0.5, 1, 0).flatten()
    acc, sen, spe=check_correct(pred, y)
    auc=metrics.roc_auc_score(y, hypo)
    
    df_hypo=df(hypo)
    df_hypo.columns=['hypothesis 1']
    
    df_pred=df(pred)
    df_pred.columns=['prediction']
    
    df_sample=df(sample)
    df_sample.columns=['sample']
    
    pred_result=pd.concat([df_sample,df_hypo, df_pred],axis=1)
    
    print("AUC : ", auc)
    
    return acc ,sen, spe ,auc, pred_result

# %%
def ML_model_performance(x, model, y, sample):
    hypo=model.predict_proba(x)[:,1]
    pred = np.where(hypo > 0.5, 1, 0).flatten()
    acc, sen, spe=check_correct(pred, y)
    auc=metrics.roc_auc_score(y, hypo)
    
    df_hypo=df(hypo)
    df_hypo.columns=['hypothesis 1']
    
    df_pred=df(pred)
    df_pred.columns=['prediction']
    
    df_sample=df(sample)
    df_sample.columns=['sample']
    
    pred_result=pd.concat([df_sample,df_hypo, df_pred],axis=1)
    
    print("AUC : ", auc)
    
    return acc ,sen, spe ,auc, pred_result


# # data loading

# In[2]:


num_of_gene = 17

deg = pd.read_csv('/nfs-data/ULMS/Diane/data_revised/GAPDH_scale_trainNDFGR_zerosum_MSE_ratio.csv')[0:num_of_gene].gene_name

tr_data = pd.read_csv('/nfs-data/ULMS/Diane/data_revised/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv')
ts_data = pd.read_csv('/nfs-data/ULMS/Diane/data_revised/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv')
theragen_data = pd.read_csv('/nfs-data/ULMS/Diane/data_revised/GAPDH_scale_trainNDFGR_zerosum_theragen.csv')

train_x=np.array(tr_data.loc[:,deg])
test_x=np.array(ts_data.loc[:,deg])

train_y=np.array(tr_data['state'])#.tolist()
test_y=np.array(ts_data['state'])#.tolist()

train_sample=tr_data['sample'].reset_index(drop = True)
test_sample=ts_data['sample'].reset_index(drop = True)

theragen1 = theragen_data.iloc[:18,:]
theragen2 = theragen_data.iloc[18:,:]

theragen1_sample = theragen1['sample'].reset_index(drop=True)
theragen2_sample = theragen2['sample'].reset_index(drop=True)

theragen1_x = np.array(theragen1.loc[:,deg])
theragen2_x = np.array(theragen2.loc[:,deg])

theragen1_y = theragen1['state'].tolist()
theragen2_y = theragen2['state'].tolist()


# In[3]:


# %%
def rf_fn(hp):
    hp_n_estimators = hp.Choice('n_estimators', [100, 200, 300, 400, 500])
    hp_max_depth = hp.Int('max_depth', min_value=5, max_value=30)
    hp_max_features = hp.Int('max_features', min_value=3, max_value=num_of_gene)
    
    model = RandomForestClassifier(n_estimators=hp_n_estimators, max_depth=hp_max_depth, max_features=hp_max_features, random_state=2021)
    
    return model

# %%
rf_tuner = tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=500,
        seed=7777),
    hypermodel=rf_fn,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    directory='/nfs-data/ULMS/Diane/model_revised/',
    project_name='RF_deg'+str(num_of_gene))

# %%
rf_tuner.search(train_x, train_y)


# In[ ]:


# %%
best_rf_hp = rf_tuner.get_best_hyperparameters(num_trials=1)[0].values
best_rf = rf_tuner.get_best_models(num_models=1)[0]

best_rf_hp_df = pd.DataFrame.from_dict([best_rf_hp])
print(best_rf_hp_df)


joblib.dump(best_rf, "/nfs-data/ULMS/Diane/model_revised/RF_deg"+str(num_of_gene)+"/GAPDH_scale_remove_race_diff_zero_sum_norm_mse_RF.h5")
best_rf_hp_df.to_csv('/nfs-data/ULMS/Diane/model_revised/RF_deg'+str(num_of_gene)+'/best_rf_hp.csv', index=False)



# In[ ]:


# %%
# best_rf = joblib.load("/home/srkim/ULMS/sample_mean_0_normalize/model/kt_bayesian_norm_MSE/RF/GAPDH_scale_remove_race_diff_zero_sum_norm_mse_RF.h5")

# %%
train_acc, train_sen, train_spe, train_auc, train_pred=ML_model_performance(train_x, best_rf, train_y, train_sample)
test_acc, test_sen, test_spe, test_auc, test_pred=ML_model_performance(test_x, best_rf, test_y, test_sample)
thera1_acc, thera1_sen, thera1_spe, thera1_auc, thera1_pred=ML_model_performance(theragen1_x, best_rf, theragen1_y, theragen1_sample)
thera2_acc, thera2_sen, thera2_spe, thera2_auc, thera2_pred=ML_model_performance(theragen2_x, best_rf, theragen2_y, theragen2_sample)


# In[ ]:




