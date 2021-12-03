import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split

df = pd.read_csv("survey_results_public.csv")
df = df[["Country","EdLevel","YearsCodePro","Employment","ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis="columns")

df = df[df["Salary"].notnull()]
# print(df.head())
# df.info()

df = df.dropna()
df.isnull().sum()

df = df[df["Salary"].notnull()]

df = df.dropna()
# print(df.isnull().sum())

df = df[df['Employment'] == 'Employed full-time']
df = df.drop("Employment", axis= 'columns')
# print(df.head(5))

l = df['Country'].value_counts()
# print(l)

def reducing_countries(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map = reducing_countries(l,400)
df['Country'] = df['Country'].map(country_map)

fig, ax = plt.subplots(1,1, figsize=(12,7))
df.boxplot('Salary','Country', ax = ax)
plt.suptitle('Salary (US$) v country')
plt.title("")
plt.ylabel("Salary")
plt.xticks(rotation=90)
# plt.show()

df = df[df['Salary']<=250000]
df = df[df['Salary']>=10000]
df = df[df['Country'] != 'Other']

# print(df['YearsCodePro'].unique())
def cleanYears(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(cleanYears)
# print(df['YearsCodePro'].unique())
# print(df['EdLevel'].unique())

def cleanEdu(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral degree' in x:
        return 'Professional degree'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(cleanEdu)
# print(df['EdLevel'].unique())
print(df.EdLevel.unique())

le_education = LabelEncoder()
le_country = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df['Country'] = le_country.fit_transform(df['Country'])

x = df.drop('Salary', axis= 1)
y = df['Salary']

X = df.drop('Salary', axis='columns')
Y = df['Salary']

modelLR = LinearRegression()
modelLR.fit(X,Y)

Y_Predict = modelLR.predict(X)
error = np.sqrt(mean_squared_error(Y,Y_Predict))
# print(error)


modelDTR = DecisionTreeRegressor(random_state=0)
modelDTR.fit(X,Y)

Y_Predict = modelDTR.predict(X)
error = np.sqrt(mean_squared_error(Y,Y_Predict))
# print(error)


# def find_best_model_using_gridsearchcv(x, y):
#     algos = {
#         'LinearRegression': {
#             'model': LinearRegression(),
#             'params': {
#                 'fit_intercept': [True, False],
#                 'copy_X': [True, False]
#             }
#         },
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [1, 10, 50, 200, 500],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#         'Ridge': {
#             'model': Ridge(),
#             'params': {
#                 'alpha': [1, 10, 50, 200, 500],
#                 'fit_intercept': [True, False]
#             }
#         },
#         'descision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion': ['squared_error', 'absolute_error'],
#                 'splitter': ['best', 'random'],
#                  ""
#             }
#         }
#     }
#     scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#     for algo_name, config in algos.items():
#         modelGS = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score=False)
#         modelGS.fit(x,y)
#         scores.append({
#             'model': algo_name,
#             'best_score': modelGS.best_score_,
#             'best_params': modelGS.best_params_
#         })
#
#     return pd.DataFrame(scores, columns=['model','best_score'])
#
# bestModel = find_best_model_using_gridsearchcv(X,Y)
# print(bestModel)

regressor = modelDTR
regressor.fit(X,Y)
Y_Predict = regressor.predict(X)
error = np.sqrt(mean_squared_error(Y,Y_Predict))
# print(f"${error}")

X = np.array([['United States',"Master’s degree", 15]])
X[:,0] = le_country.transform(X[:,0])
X[:,1] = le_education.transform(X[:,1])
X = X.astype(float)
# print(X)

Y_Predict = regressor.predict(X)
# print(Y_Predict)

import pickle
data = {'model': regressor, 'le_country': le_country, 'le_education': le_education}
with open("final_salary_prediction_model.pickle",'wb') as f:
    pickle.dump(data, f)

with open("final_salary_prediction_model.pickle",'rb') as f:
    data = pickle.load(f)

regressor_loaded = data['model']
country = data['le_country']
education = data['le_education']

Y_Predict = regressor_loaded.predict(X)
print(Y_Predict)