'''
----------------------------------------
Code and data from the Chapter 1 
----------------------------------------
'''
from turtle import pos
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
telcom = pd.read_csv('telco.csv')
# Store customerID and Churn column names
custid = ['customerID']
target = ['Churn']
# Store categorical column names
categorical = telcom.nunique()[telcom.nunique() < 5].keys().tolist()
# Remove target from the list of categorical variables
categorical.remove(target[0])
# Store numerical column names
numerical = [x for x in telcom.columns if x not in custid + target + categorical]
from sklearn.preprocessing import StandardScaler
# Perform one-hot encoding to categorical variables 
telcom = pd.get_dummies(data = telcom, columns = categorical, drop_first=True)
# Initialize StandardScaler instance
scaler = StandardScaler()
# Fit and transform the scaler on numerical columns
scaled_numerical = scaler.fit_transform(telcom[numerical])
# Build a DataFrame from scaled_numerical
scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical)
telcom[numerical]=scaled_numerical

'''
----------------------------------------
Chapter 2 - Churn prediction and drivers
----------------------------------------
'''
# Print the unique Churn values
# Calculate the ratio size of each churn group
# Import the function for splitting data to train and test
# Split the data into train and test
print(set(telcom['Churn']))
telcom.groupby(['Churn']).size() / telcom.shape[0] * 100
train, test = train_test_split(telcom, test_size = .25)

# Store column names from `telcom` excluding target variable and customer ID
cols = [col for col in telcom.columns if col not in custid + target]
# Extract training features
# Extract training target
# Extract testing features
# Extract testing target
train_X = train[cols]
train_Y = train[target]
test_X = test[cols]
test_Y = test[target]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score
logreg = LogisticRegression()

# Initialize logistic regression instance 
logreg = LogisticRegression(penalty='l1', C=0.025, solver='liblinear')
# Fit the model on training data
logreg.fit(train_X, train_Y)
# Predict churn values on test data
pred_test_Y = logreg.predict(test_X)
# Print the accuracy score on test data
print('Coeff:', np.count_nonzero(logreg.coef_))
print('Test accuracy:', round(accuracy_score(test_Y, pd.DataFrame(pred_test_Y)), 4))
print('Test recall:', round(recall_score(test_Y, pred_test_Y, pos_label = 'Yes'), 4))

C = [1, .5, .25, .1, .05, .025, .01, .005, .0025]
l1_metrics = np.zeros((len(C), 5))
l1_metrics[:,0] = C
# Run a for loop over the range of C list length
for index in range(0, len(C)):
  # Initialize and fit Logistic Regression with the C candidate
  logreg = LogisticRegression(penalty='l1', C=C[index], solver='liblinear')
  logreg.fit(train_X, train_Y)
  # Predict churn on the testing data
  pred_test_Y = logreg.predict(test_X)
  # Create non-zero count and recall score columns
  l1_metrics[index,1] = np.count_nonzero(logreg.coef_)
  l1_metrics[index,2] = accuracy_score(test_Y, pred_test_Y)
  l1_metrics[index,3] = precision_score(test_Y, pred_test_Y,pos_label='Yes')    
  l1_metrics[index,4] = recall_score(test_Y, pred_test_Y,pos_label='Yes')

# Name the columns and print the array as pandas DataFrame
col_names = ['C','Non-Zero Coeffs','Accuracy','Precision','Recall']
print(pd.DataFrame(l1_metrics, columns=col_names))
