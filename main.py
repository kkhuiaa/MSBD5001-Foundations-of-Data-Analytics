#%%
"""
Step 1:
"""


import time
import pickle as p
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import max_norm
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

from keras import regularizers

from xgboost import XGBRegressor
from script import categorytoordinal, feature_selection_by_rfe
random_state = 37 #for re-production

train0 = pd.read_csv('data/all/train.csv')
target = 'time'
test = pd.read_csv('data/all/test.csv')
samplesubmission = pd.read_csv('data/all/samplesubmission.csv')
train = train0.drop(target, axis=1)
y_train = train0[target]

def copy_data(df, times=16):
    out_df = df.copy()
    for i in range(times-1):
        out_df = df.append(out_df)
    return out_df

def create_feature(df):
    df.drop(['id', 'random_state'], axis=1, inplace=True)
    df['n_jobs'] = np.where(df['n_jobs']==-1, 10, df['n_jobs'])
    numerator_list_before = ['n_samples', 'max_iter', 'n_features', 'n_informative', 'n_classes', 'n_clusters_per_class']
    denominator_list_before = ['n_jobs', 'n_classes', 'alpha', 'n_clusters_per_class', 'penalty']
    for col in numerator_list_before+denominator_list_before:
        df[col+'_log'] = np.log(df[col]+1e-5)
        df[col+'_sqrt'] = np.sqrt(df[col])
        df[col+'_2'] = df[col]*df[col]
        df[col+'_3'] = df[col]*df[col]*df[col]
        df[col+'_4'] = df[col]*df[col]*df[col]*df[col]
        if col not in numerator_list_before:
            df[col+'_e'] = np.exp(df[col])
    numerator_list = [col for col in df.columns.tolist() if any(num in col for num in numerator_list_before)]
    denominator_list = [col for col in df.columns.tolist() if any(denom in col for denom in denominator_list_before)]
    multiply_cols = denominator_list + numerator_list
    for numerator in numerator_list:
        for denominator in denominator_list:
            df[numerator+'_over_'+col] = df[numerator]/df[denominator]
            multiply_cols.append(numerator+'_over_'+col)
        for i in range(0, len(denominator_list)-1):
            j = i + 1
            df[numerator+'_over_'+denominator_list[i]+'_'+denominator_list[j]] = df[numerator]/(df[denominator_list[i]]*df[denominator_list[j]])
            multiply_cols.append(numerator+'_over_'+denominator_list[i]+'_'+denominator_list[j])
    
    for i in range(0, len(multiply_cols)-1):
        j = i + 1
        df[multiply_cols[i]+'_multiply_'+multiply_cols[j]] = df[multiply_cols[i]]*df[multiply_cols[j]]
    for i in range(0, len(multiply_cols)-2):
        j = i + 1
        k = j + 1
        df[multiply_cols[i]+'_multiply_'+multiply_cols[j]+'_multiply_'+multiply_cols[k]] = df[multiply_cols[i]]*df[multiply_cols[j]]*df[multiply_cols[k]]
    for col in df:
        df[col] = df[col].astype(float)
    return df

cto = categorytoordinal.CategoryToOrdinal(other_threshold=1)
cto.fit(train[['penalty']], y_train)
train = train.append(copy_data(test, 16))
train_transformed = cto.transform(train)
train_transformed2 = create_feature(df=train_transformed)

sc1 = StandardScaler()
sc1.fit(train_transformed2)
train_scalez1 = pd.DataFrame(data=sc1.transform(train_transformed2), index=train_transformed2.index, columns=train_transformed2.columns)
X_train = train_scalez1[:400]
print(X_train.shape)

from genlib.ml import feature_selection_by_rfe as fsr
feature_table_rfe = fsr.feature_selection_by_rfe(X_train, y_train, [XGBRegressor]
    , cv=3, n_jobs=-1, scoring=make_scorer(mean_squared_error, greater_is_better=False)
    , random_state=random_state, plot_directory='all/evaluation/rfe/', plot_file_name='rfe_433', show_plot=True)
print(feature_table_rfe[0]) #table
print(feature_table_rfe[1]) #dict

train_rfe = train_scalez1[feature_table_rfe[0][feature_table_rfe[0]['XGBRegressor_rank'] <= 25].index.tolist()]
print(train_rfe.shape)

pca = PCA(n_components=.99)
pca.fit(train_rfe)
train_pca = pd.DataFrame(data=pca.transform(train_rfe), index=train_rfe.index, columns=['pca_'+str(i+1) for i in range(0, pca.n_components_)])

sc2 = StandardScaler()
sc2.fit(train_pca)
X = pd.DataFrame(data=sc2.transform(train_pca), index=train_pca.index, columns=train_pca.columns)
X_cols = X.columns.tolist()
X_train = X[X_cols][:400]
X_test = X[X_cols][-100:]
print(X_train.shape)

#%%
seed = random_state
def create_baseline(dropout_rate=0.3, l2=0, optimizer='Adam', activation='elu', maxnorm=5):
    model = Sequential() # create model
    model.add(Dense(30, input_dim=len(X_train.columns), activation='elu', kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(60, activation='elu', kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(45, activation='elu', kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='elu', kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(15, activation='elu', kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    return model
model = KerasRegressor(build_fn=create_baseline, batch_size=round(len(X_train)/4), epochs=500)
np.random.seed(seed)

param = dict(epochs=[500], l2=[.1, .2, .3, .4], dropout_rate=[.05, .1])

gridsearch = GridSearchCV(model, param_grid=param, cv=3, n_jobs=1, scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=0)

t0 = time.time()
gridsearch.fit(X_train, y_train, verbose=0, return_train_score=True)
t1 = time.time()

gridsearch_best = gridsearch.best_estimator_
early_stop_crit = EarlyStopping(monitor='val_loss', min_delta=0, patience=2000, verbose=0, mode='auto')
history = gridsearch_best.fit(X_train, y_train, epochs=50000, batch_size=400, verbose=0, validation_split=0.1, shuffle=True, callbacks=[early_stop_crit])

final_model = gridsearch_best
print('time:', round(t1-t0, 2)) #calculate the time for gridsearch
print('best params:', gridsearch.best_params_)

best_position = gridsearch.best_index_
print('best train score:', -gridsearch.cv_results_['mean_train_score'][best_position])
print('best train std:', gridsearch.cv_results_['std_train_score'][best_position])
print('best test score:', -gridsearch.cv_results_['mean_test_score'][best_position])
print('best test std:', gridsearch.cv_results_['std_test_score'][best_position])

train_predict = pd.Series(final_model.predict(X_train))
print('train:', train_predict.describe())
print('mse:', mean_squared_error(y_train, train_predict))

print('=================================================================================================')
print('=================================================================================================')

submission = test[['id']]
submission.loc[:, 'time_p'] = final_model.predict(X_test)
submission.loc[:, 'time_p'] = np.where(submission['time_p']<0, submission[submission['time_p']>0]['time_p'].min(), submission['time_p'])
print(submission['time_p'].describe())

submission = submission.rename(columns={'time_p': 'time'})
submission.to_csv('all/kkhuiaa_20123133_prediction_22.csv', index=False)
