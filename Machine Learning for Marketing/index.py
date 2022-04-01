import pandas as pd 
telco_raw = pd.read_csv('telco.csv')

# Print the data types of telco_raw dataset
print(telco_raw.dtypes)

# Print the header of telco_raw dataset
print(telco_raw.head())

# Print the number of unique values in each telco_raw column
print(telco_raw.nunique())