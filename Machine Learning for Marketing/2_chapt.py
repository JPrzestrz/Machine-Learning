import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
telcom = pd.read_csv('telco.csv')

# Store customerID and Churn column names
custid = ['customerID']
target = ['Churn']

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