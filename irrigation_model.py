import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb


train = pd.read_csv('train.csv') 

test = pd.read_csv('test.csv') 


train_len = len(train)
combined = pd.concat(
    [train.drop('Irrigation_Need', axis=1, errors='ignore'), test], 
    axis=0
).reset_index(drop=True)

categorical_columns = [
    'Soil_Type', 'Crop_Type',
    'Crop_Growth_Stage', 'Season', 
    'Irrigation_Type',
    'Water_Source', 'Mulching_Used', 'Region'
]

for col in categorical_columns:
    if col in combined:
        combined[col] = combined[col].astype('category')

combined['Water_Deficit'] = combined['Rainfall_mm'] + combined['Previous_Irrigation_mm'] - combined['Soil_Moisture']
combined['Temp_Moisture_Ratio'] = combined['Temperature_C']/(combined['Soil_Moisture'] + 1) 

X_train = combined.iloc[:train_len].drop(['id'], axis=1, errors='ignore')
y_train = train['Irrigation_Need']
X_test = combined.iloc[train_len:].drop(['id'], axis=1, errors='ignore')

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(
    X_test, join='left', axis=1, fill_value=0
)

params = {
    'objective': 'multiclass',
    'num_class': 3, 
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1,
    'class_weight': 'balanced' 
}

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)


train_data = lgb.Dataset(X_train, label=y_train_encoded)
model = lgb.train(
    params, train_data, num_boost_round=200
)


pred_proba = model.predict(X_test)
pred_classes_encoded = np.argmax(pred_proba, 1)
pred_classes = le.inverse_transform(pred_classes_encoded)

submission = pd.DataFrame({'id': test['id'], 'Irrigation_Need': pred_classes})

submission.to_csv('submission.csv', index=False)

print(submission.head())
