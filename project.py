!pip install optuna

from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
import xgboost as xgb
from sklearn import datasets, linear_model, metrics



df_csv = pd.read_csv('PCOS_infertility.csv')
df_excel = pd.read_excel('PCOS_data_without_infertility.xlsx', sheet_name='Full_new')

df_csv['Patient File No.'] = df_csv['Patient File No.']-10000

df_merge = pd.merge(df_csv, df_excel, on='Patient File No.',how = 'inner')
print(df_merge.to_string())

print(df_merge.to_string())

print(df_merge.shape)

#dropping UNNAMED COLUMN
df_merge.drop('Unnamed: 44', axis=1, inplace=True)

print(df_merge.shape)

df_merge.info()
df_merge.describe()


#deleting duplicates
df_merge.drop(columns=['  I   beta-HCG(mIU/mL)_y', 'II    beta-HCG(mIU/mL)_y', 'PCOS (Y/N)_y', 'AMH(ng/mL)_y'], inplace=True)

print(df_merge.shape)

print(df_merge.to_string())

df_merge.drop(columns=['Sl. No_x','Sl. No_y', 'Patient File No.'], inplace=True)

df_merge.columns

df_merge.info()

df_merge.dtypes

#need to convert all data types to numeric
for col in df_merge.columns:
    df_merge[col] = pd.to_numeric(df_merge[col], errors='coerce')

df_merge.info()

missing = df_merge.isnull().sum()
missing_percent = 100 * missing / len(df_merge)

# Plot missing values
plt.figure(figsize=(30,12))
sns.barplot(x=missing_percent.index, y=missing_percent.values)
plt.xticks(rotation=45)
plt.ylabel('Percentage of missing values')
plt.title('Missing Values by Feature')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df_merge.isnull(), cbar=False, cmap='viridis')
plt.show()

print(df_merge['II    beta-HCG(mIU/mL)_x'])

df_merge['II    beta-HCG(mIU/mL)_x'] = df_merge['II    beta-HCG(mIU/mL)_x'].fillna(df_merge['II    beta-HCG(mIU/mL)_x'].mean())
df_merge['Fast food (Y/N)'] = df_merge['Fast food (Y/N)'].fillna(df_merge['Fast food (Y/N)'].mean())
df_merge['Marraige Status (Yrs)'] = df_merge['Marraige Status (Yrs)'].fillna(df_merge['Marraige Status (Yrs)'].mean())
df_merge['AMH(ng/mL)_x'] = df_merge['AMH(ng/mL)_x'].fillna(df_merge['AMH(ng/mL)_x'].mean())

plt.figure(figsize=(12,10))
sns.heatmap(df_merge.isnull(), cbar=False, cmap='viridis')
plt.show()

data = df_merge

data.describe()

#target and feature
target = 'PCOS (Y/N)_x'
features = [col for col in data.columns if col != target]
print('Target :' , target)
print('Features :', features)

features_df = data[features]
binary_features = features_df.columns[(features_df.max()==1) & (features_df.min()==0)]

print(binary_features)

binary_data = features_df[binary_features]
data.info()
print(binary_data)

for feature_name in binary_features :
  plt.figure(figsize=(5,5))
  sns.countplot(data=data, x=feature_name,hue=target)
  plt.xlabel(feature_name)
  plt.ylabel('Count')
  plt.legend()
  plt.title(f"Countplot for {feature_name}")
  plt.show()

#statistical
#correlation matrix
corr_matrix = data.corr()
corr_matrix_sorted = corr_matrix[target].sort_values(ascending=False)
print(corr_matrix_sorted)


plt.figure(figsize=(10,8))
sns.set(font_scale=0.6)
indexes = corr_matrix.nlargest(17, target)[target].index
cm = pd.DataFrame(np.corrcoef(data[indexes].values.T))
sns.heatmap(data = cm, annot = True, yticklabels = indexes.values,
            xticklabels = indexes.values, cbar =True, cmap = 'YlGnBu',
            square = True)
plt.title(f"correlation heatmap with {target}")
plt.show()



# chi square test
x = features_df
y = data[target]

selector = SelectKBest(score_func=chi2, k=17)
X_new = selector.fit_transform(x, y)

feature_scores = selector.scores_
selected_features = x.columns[selector.get_support()]

print("Feature Scores:", feature_scores)
print("Selected Features:", selected_features)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)
print("Training set x: ",x_train)
print("Training set y: ",y_train)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print("Testing set x: ",x_test)
print("Testing set y: ",y_test)

print("Train distribution:\n", y_train.value_counts(normalize=True))
print("Test distribution:\n", y_test.value_counts(normalize=True))


selected_features = ['  I   beta-HCG(mIU/mL)_x', 'II    beta-HCG(mIU/mL)_x', 'AMH(ng/mL)_x',
       'Weight (Kg)', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH',
       'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
       'Skin darkening (Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)',
       'Follicle No. (L)', 'Follicle No. (R)']
x_train_sel = x_train[selected_features]
x_test_sel = x_test[selected_features]

#rf
def objective1(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }

    model = RandomForestClassifier(random_state=42, **params)
    model.fit(x_train_sel, y_train)
    val_preds = model.predict(x_test_sel)
    accuracy = accuracy_score(y_test, val_preds)
    return accuracy

xgb_train = xgb.DMatrix(x_train_sel, y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(x_test_sel, y_test, enable_categorical=True)

#xgboost
def objective2(trial):
 params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
 }
 n=50
 model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)
 preds = model.predict(xgb_test)
 preds = np.round(preds)
 accuracy= accuracy_score(y_test,preds)
 return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective1, n_trials=100)

print("Number of finished trials:", len(study.trials))
print("Best trial:", study.best_trial.params)
print("Best accuracy:", study.best_trial.value)

study = optuna.create_study(direction='maximize')
study.optimize(objective2, n_trials=100)

print("Number of finished trials:", len(study.trials))
print("Best trial:", study.best_trial.params)
print("Best accuracy:", study.best_trial.value)

best_rf_params = study.best_trial.params
rf_best_model = RandomForestClassifier(random_state=42, **best_rf_params)
rf_best_model.fit(x_train_sel, y_train)

# For XGBoost, since objective2 did not use trial.suggest and had fixed parameters,
# we extract those fixed parameters and train the model.
# Assuming the parameters were 'max_depth': 3, 'learning_rate': 0.1, num_boost_round=50
xgb_best_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=3, learning_rate=0.1, n_estimators=50, enable_categorical=True, random_state=42)
xgb_best_model.fit(x_train_sel, y_train)

base_learners = [
    rf_best_model,
    xgb_best_model
]

# StackingClassifier needs to be imported, if not already
from mlxtend.classifier import StackingClassifier # Assuming this was intended to be used

# Add LogisticRegression import if not already present
# from sklearn.linear_model import LogisticRegression

meta_model = linear_model.LogisticRegression(random_state=42)

# Ensure StackingClassifier is imported
stacking_model = StackingClassifier(classifiers=base_learners, meta_classifier=meta_model, use_probas=True)

model_stack = stacking_model.fit(x_train_sel, y_train)
pred_stack = model_stack.predict(x_test_sel)

acc_stack = accuracy_score(y_test, pred_stack)
print('Accuracy Score of Stacked Model:', acc_stack * 100)

import pickle
pickle.dump(model_stack, open("pcos_model.pkl", "wb"))

import os
os.getcwd()

from google.colab import files
files.download("pcos_model.pkl")
