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