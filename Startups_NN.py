#Install all the necessary packages for Implementing ANN Model
import pandas as pd
import numpy as np

# install Keras and tensorflow
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

#load the Dataset

Startup = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/50_Startups.csv")
Startup
Startup.columns = "RDS","Admin","MS","State","Profit"

# Rearrange the order of the variables
Startup = Startup.iloc[:, [4, 0, 1, 2, 3]]
Startup.columns

Startup.info()

# Covert the Categorical Variable State using Label Encoding

cat_startup = Startup.select_dtypes(include = ['object']).copy()
cat_startup.head()
print(cat_startup.isnull().values.sum()) #no null values

# if null values are there do imputation

print(cat_startup['State'].value_counts())

cat_startup_onehot_sklearn = cat_startup.copy()
cat_startup_onehot_sklearn 

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_startup_onehot_sklearn['State'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

print(lb_results_df.head())

# now concate this to the actual data sheet

startup_df = pd.concat([Startup, lb_results_df], axis=1)
startup_df
startup_df = startup_df.drop(['State'], axis=1)
startup_df

#Normalization

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

startup_n = norm_func(startup_df.iloc[:,:])

#Build the ANN Model for the Normalized data using Keras regression as the output is continuous

x = startup_n.iloc[:,1:7]
y = startup_n.iloc[:,0]
y = np.array(y)
y=np.reshape(y, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(X_train, y_train, epochs=20, batch_size=10,  verbose=1, validation_split=0.2)

print(history.history.keys())
import matplotlib.pyplot as plt
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()