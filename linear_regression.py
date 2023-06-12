
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

#loading the data set
companies=pd.read_csv('/content/drive/My Drive/1000_Companies.csv')
companies.head()

#extracting features
X=companies.iloc[:,:-1].values
#extracting targets
y=companies.iloc[:,4].values

sns.heatmap(companies.corr(),annot=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categories='auto',)
categorical_features=[3]
#onehotencoder = OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#X = X[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
print(lin_reg.coef_)
print(lin_reg.intercept_)
from sklearn.metrics import r2_score
score=r2_score(y_pred,y_test)
print(score)
