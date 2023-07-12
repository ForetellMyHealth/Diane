#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# # Data loading

# In[3]:


num_of_gene = 17

# deg = pd.read_csv('/home/anh1702/ULMS/GAPDH_scale_trainNDFGR_zerosum_MSE_cancer_normal_ratio_symbol.csv')[0:num_of_gene].id

# tr_data = pd.read_csv('/home/anh1702/ULMS/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv')
# ts_data = pd.read_csv('/home/anh1702/ULMS/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv')
# theragen_data = pd.read_csv('/home/anh1702/ULMS/GAPDH_scale_trainNDFGR_zerosum_theragen.csv')

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


# In[4]:


tf.random.set_seed(2021)
feature = deg

# %%
def model_fn(hp):
    # hp_lr = hp.Float('lr', min_value=0.0001, max_value=0.5)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.7)
    hp_layers = hp.Int('layers', min_value=2, max_value=8)
    
    input_m = Input(shape=(len(feature),))
    m_dp = Dropout(hp_dropout)(input_m)
    
    for i in range(hp_layers):
        m = Dense(hp.Int('nodes_'+str(i), min_value=2, max_value=32))(m_dp)
        m_bn = BatchNormalization()(m)
        m_dp = Activation("relu")(m_bn)
        m_dp = Dropout(hp_dropout)(m_dp)

    m_final = m_dp
    output_m = Dense(1, activation="sigmoid")(m_final)


    # adam = optimizers.Adam(lr=hp_lr)
    
    model = Model(inputs=input_m, outputs=output_m)
    model.compile(optimizer= 'adam',#'sgd', #adam, 
                    loss='binary_crossentropy',#custom_loss, 
                    metrics=['binary_crossentropy'])
    
    return model

# %%
class MyTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', min_value=8, max_value=64)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', min_value=30, max_value=200)
        #kwargs['callbacks'] = [tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=trial.hyperparameters.Int('patience', min_value=2, max_value=10))]
        
        super(MyTuner, self).run_trial(trial, *args, **kwargs)

# %%
dnn_tuner = MyTuner(
    model_fn, 
    objective= 'binary_crossentropy', #'accuracy', #'binary_crossentropy',
    max_trials=500,
    seed=7777,
    executions_per_trial=1,
    directory='/nfs-data/ULMS/Diane/model_revised/',
    project_name='DNN_deg'+str(num_of_gene))

# %%
dnn_tuner.search(train_x, train_y)


# In[5]:


# %%
# dnn_tuner.results_summary()

# %%
best_dnn_hp = dnn_tuner.get_best_hyperparameters(num_trials=1)[0].values
best_dnn = dnn_tuner.get_best_models(num_models=1)[0]

best_dnn_hp_df = pd.DataFrame.from_dict([best_dnn_hp])
print(best_dnn_hp_df)

best_dnn.save("/nfs-data/ULMS/Diane/model_revised/DNN_deg"+str(num_of_gene)+"/GAPDH_scale_remove_race_diff_zero_sum_norm_mse_DNN.h5")
best_dnn_hp_df.to_csv('/nfs-data/ULMS/Diane/model_revised/DNN_deg'+str(num_of_gene)+'/best_dnn_hp.csv', index=False)


# In[6]:


# %%
#best_dnn = load_model("/home/srkim/ULMS/sample_mean_0_normalize/model/kt_bayesian_norm_MSE/DNN/GAPDH_scale_remove_race_diff_zero_sum_norm_mse_DNN.h5")

# %%
train_acc, train_sen, train_spe, train_auc, train_pred=model_performance(train_x, best_dnn, train_y, train_sample)
test_acc, test_sen, test_spe, test_auc, test_pred=model_performance(test_x, best_dnn, test_y, test_sample)
thera1_acc, thera1_sen, thera1_spe, thera1_auc, thera1_pred=model_performance(theragen1_x, best_dnn, theragen1_y, theragen1_sample)
thera2_acc, thera2_sen, thera2_spe, thera2_auc, thera2_pred=model_performance(theragen2_x, best_dnn, theragen2_y, theragen2_sample)


# # Prediction plot

# In[ ]:


import seaborn as sns

def model_performance(x, model, y, cutoff, sample):
    hypo=model.predict(x)
    pred = np.where(hypo > cutoff, 1, 0).flatten()
    acc, sen, spe=check_correct(pred, y)
    auc=metrics.roc_auc_score(y, hypo)
    
    df_hypo=df(hypo)
    df_hypo.columns=['hypothesis 1']
    
    df_pred=df(pred)
    df_pred.columns=['prediction']
    
    df_y=df(y)
    df_y.columns=['y']
    
    df_sample=df(sample)
    df_sample.columns=['sample']
    
    pred_result=pd.concat([df_sample,df_y,df_hypo, df_pred],axis=1)
    
    print("Accuracy : ",acc)
    print("AUC : ",auc)
    print(" ")
    #print(df(pd.crosstab(y, pred, rownames=['Actual'], colnames=['Predicted'], margins=True)))
    print(" ")
    print(df(pred_result))

    return acc ,sen, spe ,auc, pred_result


# In[ ]:


# Theragen
result = model_performance(theragen1_x, best_dnn, theragen1_y, 0.5, theragen1_sample)
result[4]['state'] = np.where(theragen1_y == result[4]['prediction'], 'right', 'wrong')

data = result[4].sort_values(by=['hypothesis 1'], axis=0)
data.rename(columns = {'y' : 'type', 'hypothesis 1' : 'probability'}, inplace = True)
data.loc[data['type']==0,'type'] = 'normal'
data.loc[data['type']==1,'type'] = 'cancer'
data = data.reset_index(drop = True)

print(data)

sns.scatterplot(x = data.index, y = 'probability', hue = 'state', data = data).set_title('1st Test set')
plt.axhline(0.5, color='r')
plt.xlim(0, 18)
plt.ylim(0, 1)
#plt.savefig('/home/srkim/ULMS/sample_mean_0_normalize/result/Theragen_1st_cutoff_0311.png')


# In[ ]:


# Theragen
result = model_performance(theragen2_x, best_dnn, theragen2_y, 0.5,theragen2_sample)
result[4]['state'] = np.where(theragen2_y == result[4]['prediction'], 'right', 'wrong')


data = result[4].sort_values(by=['hypothesis 1'], axis=0)
data.rename(columns = {'y' : 'type', 'hypothesis 1' : 'probability'}, inplace = True)
data.loc[data['type']==0,'type'] = 'normal'
data.loc[data['type']==1,'type'] = 'cancer'
data = data.reset_index(drop = True)
print(data)

sns.scatterplot(x = data.index, y = 'probability', hue = 'state', data = data).set_title('2nd Test set')
plt.axhline(0.5, color='r')
plt.ylim(0, 1)
#plt.savefig('/home/srkim/ULMS/sample_mean_0_normalize/result/Theragen_2nd_cutoff_0311.png')


# In[ ]:




