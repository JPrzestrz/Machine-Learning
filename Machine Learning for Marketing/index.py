import pandas as pd 
telco_raw = pd.read_csv('telco.csv')

# Data inspection 
# Print the data types of telco_raw dataset
#print(telco_raw.dtypes)
# Print the header of telco_raw dataset
#print(telco_raw.head())
# Print the number of unique values in each telco_raw column
#print(telco_raw.nunique())

# Store customerID and Churn column names
custid = ['customerID']
target = ['Churn']
# Store categorical column names
categorical = telco_raw.nunique()[telco_raw.nunique() < 5].keys().tolist()
# Remove target from the list of categorical variables
categorical.remove(target[0])
# Store numerical column names
numerical = [x for x in telco_raw.columns if x not in custid + target + categorical]

from sklearn.preprocessing import StandardScaler
# Perform one-hot encoding to categorical variables 
telco_raw = pd.get_dummies(data = telco_raw, columns = categorical, drop_first=True)
# Initialize StandardScaler instance
scaler = StandardScaler()
# Fit and transform the scaler on numerical columns
scaled_numerical = scaler.fit_transform(telco_raw[numerical])
# Build a DataFrame from scaled_numerical
scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical)