import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\FSDS  with   GEN AI   And Agent AI\Self Material\07[03]---\emp_sal.csv")

x =dataset.iloc[:,1:2].values
y =dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(x,y)

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

plt.scatter(x, y, color= 'red')
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('Linear Regression graph')
plt.xlable('Position level')
plt.ylable('salary')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg =PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(x)

poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(x, y, color= 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title('Truth or buff(Polynomial  Regression )')
plt.xlable('Position level')
plt.ylable('salary')
plt.show()

poly_model_pred =lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred
#support  vector regression 

from sklearn.svm import SVR
svr_reg= SVR(kernel='poly',degree=6,gamma = 'auto',C =3.0, coef0=2.0)
svr_reg.fit(x,y)

svr_pred =svr_reg.predict([[6.5]])
print(svr_pred)

#KNN model

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=2)
knn_reg.fit(x, y)

#Decision tree Regression 
from sklearn.tree import DecisionTreeRegressor
dt_reg= DecisionTreeRegressor(random_state=0)
dt_reg.fit(x,y)

dt_pred =dt_reg.predict([[6.5]])
print(dt_pred)

#random Forest
from sklearn .ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=200,random_state=0,max_depth=10,min_samples_split=5,min_samples_leaf=2)
rf_reg.fit(x,y)

rf_pred=rf_reg.predict([[6.5]])
print(rf_pred)
