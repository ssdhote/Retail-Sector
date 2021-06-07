
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Importing the lifetime library for getting age of customer , frequecy, Recency and monetory value
from lifetimes.utils import summary_data_from_transaction_data



data = pd.read_excel('D:/datascience/dataset/Retail-Ecommerce.xlsx')


# In[3]:


# Droping Duplicated values
data.drop_duplicates(inplace=True)


# In[4]:


data['CustomerID'].astype('object')

new_data = data[(data['Quantity'] > 0)]

# Droping nUll values from data set
new_data.dropna(axis=0, inplace=True)


# Maximum number of customer belong to  United Kingdom.
# We wiil select United Kingdom customer only.
uk_data = new_data.query("Country == 'United Kingdom'").reset_index(drop=True)


# Slipting invoice Date and Time into date.
uk_data['invoice_date'] = uk_data['InvoiceDate'].dt.date


# Droping StockCode,Description,InvoiceDate and Country columns.
df = uk_data.drop(['StockCode', 'Description', 'InvoiceDate', 'Country'], axis=1)


# Converting invoice_date column into date time format.


df['invoice_date'] = pd.to_datetime(df['invoice_date'])


df['Totalprice'] = df['Quantity'] * df['UnitPrice']

# extracting month/year from date
df['month'] = df['invoice_date'].dt.strftime('%Y-%m')

# grouping month wise customer's total purchase
customer_data = df.groupby(['month', 'CustomerID'])['Totalprice'].sum().reset_index()


# For prediction we need only Three columns customer ID, invoice_date and Totalprice
imp_columns = df[['CustomerID', 'invoice_date', 'Totalprice']]
df2 = imp_columns


# Taking max date from data set for substracting customer first purchase date
last_oder_date = df2['invoice_date'].max()




# Puntting data into summary_data_from_transaction_data function.
lf_df = summary_data_from_transaction_data(df2, customer_id_col='CustomerID',
                                           datetime_col='invoice_date',monetary_value_col='Totalprice')


# # BetaGeoFitter Model
# Importing Betgeofitter model.
from lifetimes import BetaGeoFitter



bgf = BetaGeoFitter(penalizer_coef= 0.1)
bgf.fit(lf_df['frequency'], lf_df['recency'], lf_df['T'])


t = 30
lf_df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, lf_df['frequency'], lf_df['recency'], lf_df['T'])






from lifetimes.utils  import calibration_and_holdout_data
final_df = calibration_and_holdout_data(df, 'CustomerID', 'invoice_date',
                                           calibration_period_end ='2011-09-08',
                                          observation_period_end = '2011-12-31',
                                          freq='D').reset_index()




# # Predicting each Customer's next November months purchase count.
t = 30  #days
final_df['predicted_purchases'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, final_df['frequency_cal'],
                                                                                            final_df['recency_cal'],
                                                                                            final_df['T_cal']), 2)

final_df['pre'] = np.where(final_df['predicted_purchases'] >= 1 ,1 ,0)

import pickle
# dump final dataset in model1.pkl
pickle.dump(final_df, open('model1.pkl', 'wb'))
#dump customer dataset in model2.pkl
pickle.dump(customer_data, open('model2.pkl', 'wb'))

#load finaldataset in model1_df
model1_df = pickle.load(open('model1.pkl', 'rb'))
#load customer in model2_df
model2_df = pickle.load(open('model2.pkl', 'rb'))
