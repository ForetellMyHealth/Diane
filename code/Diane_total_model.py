# libraries
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model, Sequential 
from tensorflow.keras import backend as K

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


from tensorflow.keras import layers
import kerastuner as kt
from kerastuner import tuners


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# function
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



# Data loading

num_of_gene = 17

deg = pd.read_csv('../data_revised/GAPDH_scale_trainNDFGR_zerosum_MSE_ratio.csv')[0:num_of_gene].gene_name

tr_data = pd.read_csv('../data_revised/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv')
ts_data = pd.read_csv('../data_revised/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv')
theragen_data = pd.read_csv('../data_revised/GAPDH_scale_trainNDFGR_zerosum_theragen.csv')


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



# DNN
tf.random.set_seed(2021)
feature = deg

def model_fn(hp):
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

    
    model = Model(inputs=input_m, outputs=output_m)
    model.compile(optimizer= 'adam',
                    loss='binary_crossentropy', 
                    metrics=['binary_crossentropy'])
    
    return model

class MyTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', min_value=8, max_value=64)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', min_value=30, max_value=200)

        super(MyTuner, self).run_trial(trial, *args, **kwargs)


dnn_tuner = MyTuner(
    model_fn, 
    objective= 'binary_crossentropy',
    max_trials=500,
    seed=7777,
    executions_per_trial=1,
    directory='../model/',
    project_name='DNN_deg'+str(num_of_gene))


dnn_tuner.search(train_x, train_y)


best_dnn_hp = dnn_tuner.get_best_hyperparameters(num_trials=1)[0].values
best_dnn = dnn_tuner.get_best_models(num_models=1)[0]

best_dnn_hp_df = pd.DataFrame.from_dict([best_dnn_hp])
print(best_dnn_hp_df)


# DNN result
train_acc, train_sen, train_spe, train_auc, train_pred=model_performance(train_x, best_dnn, train_y, train_sample)
test_acc, test_sen, test_spe, test_auc, test_pred=model_performance(test_x, best_dnn, test_y, test_sample)
thera1_acc, thera1_sen, thera1_spe, thera1_auc, thera1_pred=model_performance(theragen1_x, best_dnn, theragen1_y, theragen1_sample)
thera2_acc, thera2_sen, thera2_spe, thera2_auc, thera2_pred=model_performance(theragen2_x, best_dnn, theragen2_y, theragen2_sample)





# Gradient Boosting
def gb_fn(hp):
    hp_lr = hp.Float('lr', min_value=0.0001, max_value=0.5)
    hp_n_estimators = hp.Choice('n_estimators', [100, 200, 300, 400, 500])
    hp_max_depth = hp.Int('max_depth', min_value=5, max_value=30)
    hp_max_features = hp.Int('max_features', min_value=3, max_value=num_of_gene)
    
    model = GradientBoostingClassifier(learning_rate=hp_lr, n_estimators=hp_n_estimators, max_depth=hp_max_depth, 
                                       max_features=hp_max_features, random_state=2021)
    
    return model

gb_tuner = tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=500,
        seed=7777),
    hypermodel=gb_fn,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    directory='../model/',
    project_name='GB_deg'+str(num_of_gene))

gb_tuner.search(train_x, train_y)

best_gb_hp = gb_tuner.get_best_hyperparameters(num_trials=1)[0].values
best_gb = gb_tuner.get_best_models(num_models=1)[0]

best_gb_hp_df = pd.DataFrame.from_dict([best_gb_hp])
print(best_gb_hp_df)


# Gradient Boosting result
train_acc, train_sen, train_spe, train_auc, train_pred=ML_model_performance(train_x, best_gb, train_y, train_sample)
test_acc, test_sen, test_spe, test_auc, test_pred=ML_model_performance(test_x, best_gb, test_y, test_sample)
thera1_acc, thera1_sen, thera1_spe, thera1_auc, thera1_pred=ML_model_performance(theragen1_x, best_gb, theragen1_y, theragen1_sample)
thera2_acc, thera2_sen, thera2_spe, thera2_auc, thera2_pred=ML_model_performance(theragen2_x, best_gb, theragen2_y, theragen2_sample)





# Random Forest
def rf_fn(hp):
    hp_n_estimators = hp.Choice('n_estimators', [100, 200, 300, 400, 500])
    hp_max_depth = hp.Int('max_depth', min_value=5, max_value=30)
    hp_max_features = hp.Int('max_features', min_value=3, max_value=num_of_gene)
    
    model = RandomForestClassifier(n_estimators=hp_n_estimators, max_depth=hp_max_depth, max_features=hp_max_features, random_state=2021)
    
    return model

rf_tuner = tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=500,
        seed=7777),
    hypermodel=rf_fn,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    directory='../model/',
    project_name='RF_deg'+str(num_of_gene))


rf_tuner.search(train_x, train_y)


best_rf_hp = rf_tuner.get_best_hyperparameters(num_trials=1)[0].values
best_rf = rf_tuner.get_best_models(num_models=1)[0]

best_rf_hp_df = pd.DataFrame.from_dict([best_rf_hp])
print(best_rf_hp_df)

# Random Forest result
train_acc, train_sen, train_spe, train_auc, train_pred=ML_model_performance(train_x, best_rf, train_y, train_sample)
test_acc, test_sen, test_spe, test_auc, test_pred=ML_model_performance(test_x, best_rf, test_y, test_sample)
thera1_acc, thera1_sen, thera1_spe, thera1_auc, thera1_pred=ML_model_performance(theragen1_x, best_rf, theragen1_y, theragen1_sample)
thera2_acc, thera2_sen, thera2_spe, thera2_auc, thera2_pred=ML_model_performance(theragen2_x, best_rf, theragen2_y, theragen2_sample)




# SVM
def svm_fn(hp):
    hp_kernel = hp.Choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    hp_C = hp.Choice('C', [0.01, 0.1, 1.0, 10.0, 100.0])
    hp_gamma = hp.Choice('gamma', [0.001, 0.01, 0.1, 0.5, 1.0, 10.0])
    
    model = SVC(C=hp_C, kernel=hp_kernel, gamma=hp_gamma, probability=True, random_state=2021)
    
    return model

svm_tuner = tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=500,
        seed=7777),
    hypermodel=svm_fn,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    directory='../model/',
    project_name='SVM_deg'+str(num_of_gene))

svm_tuner.search(train_x, train_y)

best_svm_hp = svm_tuner.get_best_hyperparameters(num_trials=1)[0]
best_svm = svm_tuner.get_best_models(num_models=1)[0]

best_svm_hp_df = pd.DataFrame.from_dict([best_svm_hp])
print(best_svm_hp_df)

# SVM result
train_acc, train_sen, train_spe, train_auc, train_pred=ML_model_performance(train_x, best_svm, train_y, train_sample)
test_acc, test_sen, test_spe, test_auc, test_pred=ML_model_performance(test_x, best_svm, test_y, test_sample)
thera1_acc, thera1_sen, thera1_spe, thera1_auc, thera1_pred=ML_model_performance(theragen1_x, best_svm, theragen1_y, theragen1_sample)
thera2_acc, thera2_sen, thera2_spe, thera2_auc, thera2_pred=ML_model_performance(theragen2_x, best_svm, theragen2_y, theragen2_sample)
