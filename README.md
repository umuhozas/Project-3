# Project3

This presentation includes the concepts of multivariate regression analysis and gradient boosting. For this project I am going to apply lowess regresssion method, boosted lowess, and compare their performance to other regression methods like Random Forest and Neural Networks. For each method, I am going to calculate its cross validated mean square error and mean absolute error to conclude which one is better.

Using Multivariable regression will help us to identify the reationships between dependent and multiple independent variables. I am going to use the cars.csv dataset and I will use indepedent variable X = cars[['ENG','CYL','WGT']].values and dependent variable y = cars['MPG'].values. In addition, we use multivariate regression to predict behavior of the outcome variable and how they change.

Before we start our analysis, we are going to import different libraries that we are going to use in reading and manipulating our data.
```Python
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
from matplotlib import pyplot

```
Additionally, we also import libraries to create a Neural Network
```Python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam # they recently updated Tensorflow
from keras.callbacks import EarlyStopping
```

Next, I downloaded the kernels to help in constructing the non-linear decision boundaries using linear classifiers. 

```Python
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
```

```Python
#Locally Weighted Regressor

I created a Locally Weighted Regression Function that performs a regression around a point of interest using only training data that are local to that point. This locally weighted regression function takes the independent, dependent value, the choice of kernel, and hyperparameter tau. In this function, we expect x to be sorted in an increasing order

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```
```Python
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
```
I am going to import xgboost because It tells us about the difference between actual values and predicted values. It tells us how the results we have are related to the real values. The xgb function also analyzes the complexity of the model. It penalizes complex models through both LASSO (L1) and Ridge (L2) regularization to prevent overfitting. The ultimate goal is to find simple and accurate models. xgboost uses regularization parameters like gamma, alpha, and lambda. 


```Python
import xgboost as xgb
```

```Python
# we design a Neural Network for regression
# We have to decide how many layers we want, how many neurons per layer and the type of activation functions
# Create a Neural Network model
model_nn = Sequential()
model_nn.add(Dense(128, activation="relu", input_dim=3))
model_nn.add(Dense(128, activation="relu"))
#model_nn.add(Dense(128, activation="relu"))
#model_nn.add(Dense(128, activation="relu"))
#model_nn.add(Dense(128, activation ="relu"))
# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model_nn.add(Dense(1, activation="linear")) # we need this b/c we predict a continuous random variable

# Compile model: The model is initialized with the Adam optimizer and then it is compiled.
model_nn.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-2)) # lr=1e-3, decay=1e-3 / 200)

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)
```

```Python
import lightgbm as lgb
```
Here, I selected features to use as my dependent and independent variable.

```Python
#Let's import our data from google drive
cars = pd.read_csv('drive/MyDrive/Colab Notebooks/Data_410/data/cars.csv')
X = cars[['ENG','CYL','WGT']].values
y = cars['MPG'].values
```
In the code below, I selected the regression analysis method hypothesus to use in calculating the mean absolute errors and mean squared error.

```Python
mse_lwr = []
mae_lwr = []
mse_blwr = []
mae_blwr =[]
mse_rf = []
mae_rf = []
mse_xgb = []
mae_xgb = []
mse_nn = []
mae_nn = []
mse_nw = []
mae_nw = []

for i in [1234]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  scale = StandardScaler()
# this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
  dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=1,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=4)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  model_KernReg = KernelReg(endog=dat_train[:,-1],exog=dat_train[:,:-1],var_type='ccc',ckertype='gaussian')
  yhat_sm, yhat_std = model_KernReg.fit(dat_test[:,:-1])
  model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
  yhat_nn = model_nn.predict(xtest)
  #model_nn.fit(xtrain,ytrain,validation_split=0.3, epochs=500, batch_size=20, verbose=0, callbacks=[es])
  #yhat_nn = model_nn.predict(xtest)
  mse_lwr.append(mse(ytest, yhat_lwr))
  mae_lwr.append(mean_absolute_error(ytest, yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mae_blwr.append(mean_absolute_error(ytest, yhat_blwr))
  mse_rf.append(mse(ytest,yhat_rf))
  mae_rf.append(mean_absolute_error(ytest, yhat_rf))
  mse_xgb.append(mse(ytest,yhat_xgb))
  mae_xgb.append(mean_absolute_error(ytest, yhat_xgb))
  mse_nn.append(mse(ytest,yhat_nn))
  mae_nn.append(mean_absolute_error(ytest,yhat_nn))
  mse_nw.append(mse(ytest,yhat_sm))
  mae_nw.append(mean_absolute_error(ytest,yhat_sm))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Absolute Error for LWR is : '+str(np.mean(mae_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Absolute Error for BLWR is : '+str(np.mean(mae_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Absolute Error for RF is : '+str(np.mean(mae_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Absolute Error for XGB is : '+str(np.mean(mae_xgb)))
print('The Cross-validated Mean Squared Error for NN is : '+str(np.mean(mse_nn)))
print('The Cross-validated Mean Absolute Error for NN is : '+str(np.mean(mae_nn)))
print('The Cross-validated Mean Squared Error for NW is : '+str(np.mean(mse_nw)))
print('The Cross-validated Mean Absolute Error for NW is : '+str(np.mean(mae_nw)))
```

The Cross-validated Mean Squared Error for LWR is : 16.87007492097876

The Cross-validated Mean Absolute Error for LWR is : 2.982604175901108

The Cross-validated Mean Squared Error for BLWR is : 16.7211579313579

The Cross-validated Mean Absolute Error for BLWR is : 2.937089045036899

The Cross-validated Mean Squared Error for RF is : 16.73276818715321

The Cross-validated Mean Absolute Error for RF is : 2.9485279800717104

The Cross-validated Mean Squared Error for XGB is : 15.96043146844057

The Cross-validated Mean Absolute Error for XGB is : 2.888928138635418

The Cross-validated Mean Squared Error for NN is : 20.79736161414629

The Cross-validated Mean Absolute Error for NN is : 3.1242736013596013

The Cross-validated Mean Squared Error for NW is : 17.244727908363686

The Cross-validated Mean Absolute Error for NW is : 3.0559279028620145


```Python
mse_lwr = []
mae_lwr = []
mse_blwr = []
mae_blwr =[]
mse_rf = []
mae_rf = []
mse_xgb = []
mae_xgb = []
mse_nn = []
mae_nn = []
mse_nw = []
mae_nw = []

for i in [1234]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  scale = StandardScaler()
# this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
  dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=1,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  model_KernReg = KernelReg(endog=dat_train[:,-1],exog=dat_train[:,:-1],var_type='ccc',ckertype='gaussian')
  yhat_sm, yhat_std = model_KernReg.fit(dat_test[:,:-1])
  model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=15, verbose=0, callbacks=[es])
  yhat_nn = model_nn.predict(xtest)
  #model_nn.fit(xtrain,ytrain,validation_split=0.3, epochs=500, batch_size=20, verbose=0, callbacks=[es])
  #yhat_nn = model_nn.predict(xtest)
  mse_lwr.append(mse(ytest, yhat_lwr))
  mae_lwr.append(mean_absolute_error(ytest, yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mae_blwr.append(mean_absolute_error(ytest, yhat_blwr))
  mse_rf.append(mse(ytest,yhat_rf))
  mae_rf.append(mean_absolute_error(ytest, yhat_rf))
  mse_xgb.append(mse(ytest,yhat_xgb))
  mae_xgb.append(mean_absolute_error(ytest, yhat_xgb))
  mse_nn.append(mse(ytest,yhat_nn))
  mae_nn.append(mean_absolute_error(ytest,yhat_nn))
  mse_nw.append(mse(ytest,yhat_sm))
  mae_nw.append(mean_absolute_error(ytest,yhat_sm))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Absolute Error for LWR is : '+str(np.mean(mae_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Absolute Error for BLWR is : '+str(np.mean(mae_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Absolute Error for RF is : '+str(np.mean(mae_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Absolute Error for XGB is : '+str(np.mean(mae_xgb)))
print('The Cross-validated Mean Squared Error for NN is : '+str(np.mean(mse_nn)))
print('The Cross-validated Mean Absolute Error for NN is : '+str(np.mean(mae_nn)))
print('The Cross-validated Mean Squared Error for NW is : '+str(np.mean(mse_nw)))
print('The Cross-validated Mean Absolute Error for NW is : '+str(np.mean(mae_nw)))
```

The Cross-validated Mean Squared Error for LWR is : 16.87007492097876

The Cross-validated Mean Absolute Error for LWR is : 2.982604175901108

The Cross-validated Mean Squared Error for BLWR is : 16.7211579313579

The Cross-validated Mean Absolute Error for BLWR is : 2.937089045036899

The Cross-validated Mean Squared Error for RF is : 16.4866476278051

The Cross-validated Mean Absolute Error for RF is : 2.9468332649602815

The Cross-validated Mean Squared Error for XGB is : 15.96043146844057

The Cross-validated Mean Absolute Error for XGB is : 2.888928138635418

The Cross-validated Mean Squared Error for NN is : 19.81132299195614

The Cross-validated Mean Absolute Error for NN is : 2.875250464406089

The Cross-validated Mean Squared Error for NW is : 17.244727908363686

The Cross-validated Mean Absolute Error for NW is : 3.0559279028620145


```Python
mse_lwr = []
mae_lwr = []
mse_blwr = []
mae_blwr =[]
mse_rf = []
mae_rf = []
mse_xgb = []
mae_xgb = []
mse_nn = []
mae_nn = []
mse_nw = []
mae_nw = []

for i in [410]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  scale = StandardScaler()
# this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
  dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=1,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=4)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  model_KernReg = KernelReg(endog=dat_train[:,-1],exog=dat_train[:,:-1],var_type='ccc',ckertype='gaussian')
  yhat_sm, yhat_std = model_KernReg.fit(dat_test[:,:-1])
  model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
  yhat_nn = model_nn.predict(xtest)
  #model_nn.fit(xtrain,ytrain,validation_split=0.3, epochs=500, batch_size=20, verbose=0, callbacks=[es])
  #yhat_nn = model_nn.predict(xtest)
  mse_lwr.append(mse(ytest, yhat_lwr))
  mae_lwr.append(mean_absolute_error(ytest, yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mae_blwr.append(mean_absolute_error(ytest, yhat_blwr))
  mse_rf.append(mse(ytest,yhat_rf))
  mae_rf.append(mean_absolute_error(ytest, yhat_rf))
  mse_xgb.append(mse(ytest,yhat_xgb))
  mae_xgb.append(mean_absolute_error(ytest, yhat_xgb))
  mse_nn.append(mse(ytest,yhat_nn))
  mae_nn.append(mean_absolute_error(ytest,yhat_nn))
  mse_nw.append(mse(ytest,yhat_sm))
  mae_nw.append(mean_absolute_error(ytest,yhat_sm))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Absolute Error for LWR is : '+str(np.mean(mae_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Absolute Error for BLWR is : '+str(np.mean(mae_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Absolute Error for RF is : '+str(np.mean(mae_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Absolute Error for XGB is : '+str(np.mean(mae_xgb)))
print('The Cross-validated Mean Squared Error for NN is : '+str(np.mean(mse_nn)))
print('The Cross-validated Mean Absolute Error for NN is : '+str(np.mean(mae_nn)))
print('The Cross-validated Mean Squared Error for NW is : '+str(np.mean(mse_nw)))
print('The Cross-validated Mean Absolute Error for NW is : '+str(np.mean(mae_nw)))
```

The Cross-validated Mean Squared Error for LWR is : 17.099342425016683

The Cross-validated Mean Absolute Error for LWR is : 3.0069815242629048

The Cross-validated Mean Squared Error for BLWR is : 17.94330086451972

The Cross-validated Mean Absolute Error for BLWR is : 3.056783299853078

The Cross-validated Mean Squared Error for RF is : 17.17922265995678

The Cross-validated Mean Absolute Error for RF is : 3.0033395275225674

The Cross-validated Mean Squared Error for XGB is : 16.30102036531247

The Cross-validated Mean Absolute Error for XGB is : 2.9209418791234416
The Cross-validated Mean Squared Error for NN is : 19.103377219838517
The Cross-validated Mean Absolute Error for NN is : 2.9168079546614347
The Cross-validated Mean Squared Error for NW is : 18.173336639961313
The Cross-validated Mean Absolute Error for NW is : 3.0939539828184883
