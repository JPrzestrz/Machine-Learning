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

# Build a pivot table counting invoices for each customer monthly
cust_month_tx = pd.pivot_table(data=online, values='InvoiceNo',
                               index=['CustomerID'], columns=['InvoiceMonth'],
                               aggfunc=pd.Series.nunique, fill_value=0)
# Store November 2011 data column name as a list
target = ['2011-11']
# Store target value as `Y`
Y = cust_month_tx[target]

# Store customer identifier column name as a list
custid = ['CustomerID']
# Select feature column names excluding customer identifier
cols = [col for col in features.columns if col not in custid]
# Extract the features as `X`
X = features[cols]
# Split data to training and testing
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=99)

from sklearn.linear_model import LinearRegression
# Initialize linear regression instance
linreg = LinearRegression()
# Fit the model to training dataset
linreg.fit(train_X, train_Y)
# Predict the target variable for training data
train_pred_Y = linreg.predict(train_X)
# Predict the target variable for testing data
test_pred_Y = linreg.predict(test_X)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Calculate root mean squared error on training data
rmse_train = np.sqrt(mean_squared_error(train_Y, train_pred_Y))
# Calculate mean absolute error on training data
mae_train = mean_absolute_error(train_Y, train_pred_Y)
# Calculate root mean squared error on testing data
rmse_test = np.sqrt(mean_squared_error(test_Y, test_pred_Y))
# Calculate mean absolute error on testing data
mae_test = mean_absolute_error(test_Y, test_pred_Y)
# Print the performance metrics
print('RMSE train: {}; RMSE test: {}\nMAE train: {}, MAE test: {}'.format(rmse_train, rmse_test, mae_train, mae_test))

# Import `statsmodels.api` module
import statsmodels.api as sm
# Initialize model instance on the training data
olsreg = sm.OLS(train_Y, train_X)
# Fit the model
olsreg = olsreg.fit()
# Print model summary
print(olsreg.summary())

''' 
output:
                                OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.488
Model:                            OLS   Adj. R-squared (uncentered):              0.487
Method:                 Least Squares   F-statistic:                              480.3
Date:                Wed, 13 Apr 2022   Prob (F-statistic):                        0.00
Time:                        15:09:19   Log-Likelihood:                         -2769.8
No. Observations:                2529   AIC:                                      5550.
Df Residuals:                    2524   BIC:                                      5579.
Df Model:                           5                                                  
Covariance Type:            nonrobust                                                  
==================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
recency            0.0002      0.000      1.701      0.089   -2.92e-05       0.000
frequency          0.1316      0.003     38.000      0.000       0.125       0.138
monetary        1.001e-06   3.59e-05      0.028      0.978   -6.95e-05    7.15e-05
quantity_avg       0.0001      0.000      0.803      0.422      -0.000       0.000
quantity_total    -0.0001   5.74e-05     -2.562      0.010      -0.000   -3.45e-05
==============================================================================
Omnibus:                      987.494   Durbin-Watson:                   1.978
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5536.657
Skew:                           1.762   Prob(JB):                         0.00
Kurtosis:                       9.334   Cond. No.                         249.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''