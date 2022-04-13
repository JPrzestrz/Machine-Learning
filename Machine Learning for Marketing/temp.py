'''
----------------------------------------------------
Chapter 3 - Customer Lifetime Value (CLV) prediction
----------------------------------------------------
'''
import pandas as pd
import numpy as np
import datetime as dt
cohort_counts=pd.read_fwf('retention.txt')

# Extract cohort sizes from the first column of cohort_counts
cohort_sizes = cohort_counts.iloc[:,0]
print(cohort_counts.head())
# Calculate retention by dividing the counts with the cohort sizes
retention = cohort_counts.divide(cohort_sizes, axis=0)
# Calculate churn
churn = 1 - retention
# Print the retention table
print(retention)
# Calculate the mean retention rate
retention_rate = retention.iloc[:,1:].mean().mean()
# Calculate the mean churn rate
churn_rate = churn.iloc[:,1:].mean().mean()
# Print rounded retention and churn rates
print('Retention rate: {:.2f}; Churn rate: {:.2f}'.format(retention_rate, churn_rate))

# Calculate monthly spend per customer
monthly_revenue = online.groupby(['CustomerID','InvoiceMonth'])['TotalSum'].sum()
# Calculate average monthly spend
monthly_revenue = np.mean(monthly_revenue)
# Define lifespan to 36 months
lifespan_months = 36
# Calculate basic CLV
clv_basic = monthly_revenue * lifespan_months
# Print the basic CLV value
print('Average basic CLV is {:.1f} USD'.format(clv_basic))

# Calculate average revenue per invoice
revenue_per_purchase = online.groupby(['InvoiceNo'])['TotalSum'].mean().mean()
# Calculate average number of unique invoices per customer per month
frequency_per_month = online.groupby(['CustomerID','InvoiceMonth'])['InvoiceNo'].nunique().mean()
# Define lifespan to 36 months
lifespan_months = 36
# Calculate granular CLV
clv_granular = revenue_per_purchase * frequency_per_month * lifespan_months
# Print granular CLV value
print('Average granular CLV is {:.1f} USD'.format(clv_granular))

# Calculate monthly spend per customer
monthly_revenue = online.groupby(['CustomerID','InvoiceMonth'])['TotalSum'].sum().mean()
# Calculate average monthly retention rate
retention_rate = retention.iloc[:,1:].mean().mean()
# Calculate average monthly churn rate
churn_rate = 1 - retention_rate
# Calculate traditional CLV 
clv_traditional = monthly_revenue * (retention_rate / churn_rate)
# Print traditional CLV and the retention rate values
print('Average traditional CLV is {:.1f} USD at {:.1f} % retention_rate'.format(clv_traditional, retention_rate*100))

# Define the snapshot date
NOW = dt.datetime(2011,11,1)

# Calculate recency by subtracting current date from the latest InvoiceDate
features = online_X.groupby('CustomerID').agg({
  'InvoiceDate': lambda x: (NOW - x.max()).days,
  # Calculate frequency by counting unique number of invoices
  'InvoiceNo': pd.Series.nunique,
  # Calculate monetary value by summing all spend values
  'TotalSum': np.sum,
  # Calculate average and total quantity
  'Quantity': ['mean', 'sum']}).reset_index()

# Rename the columns
features.columns = ['CustomerID', 'recency', 'frequency', 'monetary', 'quantity_avg', 'quantity_total']