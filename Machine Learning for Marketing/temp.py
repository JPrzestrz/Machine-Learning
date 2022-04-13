'''
----------------------------------------------------
Chapter 3 - Customer Lifetime Value (CLV) prediction
----------------------------------------------------
'''
import pandas as pd
import numpy as np
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