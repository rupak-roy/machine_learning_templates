

#Multiple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('abalone.data',header=None)
col_names = [['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings']]	
dataset.columns = col_names

X = dataset.iloc[:,:-1].values

#All columns except the last column (by defining the upper bound)
y = dataset.iloc[:, 8].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Gender column
ct = ColumnTransformer([("Gender", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

#to avoid dummy variable trap
X = X[:, 1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#if we wish to predict by manually entering the values then we have #to put number of values = number of columns, representing each #value to its corresponding column
regressor.predict([[1.0,1.0,0.55,0.45,0.15,0.91,0.277,0.243,0.33]])

#To get the intercept:
print(regressor.intercept_)

#To view the coefficient values
print(regressor.coef_)

#We can also compare the actual versus predicted
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

#we can also visualize the actual vs predicted
df_1 = df.head(25)
df_1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#evaluation Metrics 
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#save the model in the disk
import pickle
# save the model to disk
filename = 'reg_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

#load the model from disk
filename1 = 'reg_model.sav'
loaded_model = pickle.load(open(filename1, 'rb'))

#another method using joblib
'''Pickled model as a file using joblib: Joblib is the replacement of pickle as it is more efficent on objects that carry large numpy arrays.'''
from sklearn.externals import joblib 

#Save the model as a pickle in a file 
joblib.dump(regressor, 'regressor.pkl') 
  
#Load the model from the file 
loaded_model2 = joblib.load('regressor.pkl')  
  
#Use the loaded model to make predictions 
loaded_model2.predict(X_test)
##############################################################


#Polynomial Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('abalone.data', header = None)
col_names = [['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings']]	
dataset.columns = col_names

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Gender column
ct = ColumnTransformer([("Gender", OneHotEncoder(), [0])], remainder = 'passthrough')
dataset = ct.fit_transform(dataset)
"""#anyway we wont use this column we will simply use 1 independent variable i.e. column 'Length' of abalone  data set and X our dependent variable  i.e.'number of rings' from the last column.
The reason why i m choosing only 2 columns is to show u the comparison of  performance of both the algorithm using plots()"""

X = dataset[:,10:]
y = dataset[:,3]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

#To view the X-plot features
X_poly

#build up the regression(poly) model
poly_reg.fit(X_poly, y) 
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_reg.predict(X), color = 'blue')
plt.title('Predicting abalone from physical measurements.')
plt.xlabel('Rings')
plt.ylabel('Length')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Predicting abalone from physical measurements.')
plt.xlabel('Rings')
plt.ylabel('length')
plt.show()

# Visualizing the Polynomial Regression results (for smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Predicting abalone from physical measurements.')
plt.xlabel('Rings')
plt.ylabel('Length')
plt.show()

# Predicting a new result with Linear Regression
#linear_reg.predict([length])
linear_reg.predict([[10]])

# Predicting a new result with Polynomial Regression
linear_reg2.predict(poly_reg.fit_transform([[10]]))


##################################################################
#SVR
# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset 
data = pd.read_csv('AirPressure.csv') 
data

#dividing the dataset into X and y
X = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

#Feature Scaling for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_y.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))

#Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression()
lin.fit(X, y)

#Fitting SVR to the dataset 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Visualizing the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

# Visualizing the SVR results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, regressor.predict(X), color = 'red') 
plt.title('Support Vector Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

#Predicting a new result with Linear Regression 
lin.predict([[150.0]])

#Predicting a new result(pressure) with Support Vector Regression
y_Pressure = regressor.predict([[55]])

#########################################################
#Decision Trees

# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset 
data = pd.read_csv('AirPressure.csv') 
data

#dividing the dataset into X and y
X = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

#Feature Scaling for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_y.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))

#Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression()
lin.fit(X, y)

#Fitting SVR to the dataset 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Fitting Decision Tree Regression 
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state = 0)
dt_model.fit(X, y)

#Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

#Visualising the SVR results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, regressor.predict(X), color = 'red') 
plt.title('Support Vector Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

#Visualising the Decision Trees Regression results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, dt_model.predict(X), color = 'red') 
plt.title('Decision Trees Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

#we will see the line is passing between the blue points(Thus better model)
#Predicting a new result with Linear Regression 
lin.predict([[55.0]])

#Predicting a new result(pressure) with Support Vector Regression
y_Pressure = dt_model.predict([[55]])

from sklearn.tree import export_graphviz  
#export the decision tree to a tree.dot file 
#for visualizing the plot easily anywhere 
export_graphviz(dt_model, out_file ='e:/tree.dot', 
               feature_names =['Pressure'])
"""
The tree is finally exported and we can visualized using  
http://www.webgraphviz.com/ by copying the data from the 'tree.dot' file."""

import pickle
#save the model to disk
filename = 'final_model.sav'
pickle.dump(dt_model, open(filename, 'wb'))
#load the model from disk
filename1 = 'final_model.sav'
loaded_model = pickle.load(open(filename1, 'rb'))

#another method using joblib
'''Pickled model as a file using joblib: Joblib is the replacement of pickle as
 it is more efficent on objects that carry large numpy arrays. 
'''
from sklearn.externals import joblib 
#Save the model as a pickle in a file 
joblib.dump(dt_model, 'dt_model.pkl') 
  
#Load the model from the file 
loaded_model2 = joblib.load('dt_model.pkl')  
  
#Use the loaded model to make predictions 
loaded_model2.predict([[55]])

###########################################
#Random Forest
#Importing the libraries 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#Importing the dataset 
data = pd.read_csv('AirPressure.csv') 
data

#dividing the dataset into X and y
X = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

#Feature Scaling for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_y.fit_transform(X.reshape(-1,1))
#X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#Fitting Decision Tree Regression 
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state = 0)
dt_model.fit(X, y)
 
#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 500, random_state = 0)
rf_model.fit(X, y)

# Visualizing the Decision Trees Regression results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, dt_model.predict(X), color = 'red') 
plt.title('Decision Trees Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

# Visualizing the Random Forest results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, rf_model.predict(X), color = 'red') 
plt.title('Random Forest Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show()

# Visualizing the Random Forest results in high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, rf_model.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

#Predicting a new result(pressure) with Random Forest Regression 
rf_model.predict([[55]])
#Predicting a new result(pressure) with Decision Tree Regression
dt_model.predict([[55]])