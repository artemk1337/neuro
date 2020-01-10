import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
import pandas as pd
import csv


df = pd.read_csv('vgsales.csv')
print(df.head())

y = df['Global_Sales']

df = df.drop(['Rank', 'Global_Sales', 'Name', 'Platform', 'Genre', 'Publisher'], axis=1)
X = df.get_values()
X = np.nan_to_num(X)

y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.25)

model_reg = LinearRegression()

model_reg.fit(X_train, y_train)
y_pred_reg = model_reg.predict(X_test)
print(y_pred_reg)
print(mae(y_test, y_pred_reg))

plt.scatter(y_test, y_pred_reg)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.axis('equal')
plt.axis('square')
plt.show()

print(model_reg.coef_)






