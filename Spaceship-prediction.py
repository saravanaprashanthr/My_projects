#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install pandas numpy optuna xgboost scikit-learn')


# In[4]:


get_ipython().system('pip install scikit-learn')


# In[1]:


import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# In[2]:


train_df = pd.read_csv('spaceship-titanic/train.csv')
test_df = pd.read_csv('spaceship-titanic/test.csv')


# In[9]:


# 2. Feature Engineering
def engineer_features(df):
    df = df.copy()
    # Split Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    # Expenditure features
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[exp_cols].sum(axis=1)
    df['NoSpending'] = (df['TotalSpent'] == 0).astype(int)
    # Drop columns not useful for prediction
    return df.drop(columns=['Name', 'Cabin', 'Num'])

# We keep PassengerId separate for the final submission file
train_proc = engineer_features(train_df).drop(columns=['PassengerId'])
test_ids = test_df['PassengerId']
test_proc = engineer_features(test_df).drop(columns=['PassengerId'])


# In[10]:


# 3. Handle Missing Values & Encoding
for col in train_proc.select_dtypes(include=['object']).columns:
    if col != 'Transported':
        # Fill missing with mode
        fill_val = train_proc[col].mode()[0]
        train_proc[col] = train_proc[col].fillna(fill_val)
        test_proc[col] = test_proc[col].fillna(fill_val)
        
        # Encode
        le = LabelEncoder()
        # Fit on combined data to ensure all categories are covered
        full_data = pd.concat([train_proc[col], test_proc[col]], axis=0)
        le.fit(full_data)
        train_proc[col] = le.transform(train_proc[col])
        test_proc[col] = le.transform(test_proc[col])

# Fill numeric NaNs
for col in train_proc.select_dtypes(include=['float64']).columns:
    med = train_proc[col].median()
    train_proc[col] = train_proc[col].fillna(med)
    test_proc[col] = test_proc[col].fillna(med)


# In[11]:


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    clf = XGBClassifier(**params, eval_metric='logloss')
    # Using cross-validation to ensure model robustness
    score = cross_val_score(clf, train_proc.drop('Transported', axis=1), 
                            train_proc['Transported'].astype(int), cv=3).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)


# In[12]:


best_model = XGBClassifier(**study.best_params, eval_metric='logloss')
best_model.fit(train_proc.drop('Transported', axis=1), train_proc['Transported'].astype(int))

predictions = best_model.predict(test_proc)

# Convert 0/1 back to False/True for submission
submission = pd.DataFrame({
    'PassengerId': test_ids, 
    'Transported': predictions.astype(bool)
})
submission.to_csv('my_submission.csv', index=False)
print("Submission saved successfully as 'my_submission.csv'!")


# In[ ]:




