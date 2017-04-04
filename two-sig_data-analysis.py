# Data analysis and processing for Two Sigma Rental Listing competition
# Author: Jeremy Melamed
# Date: April, 2017

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Read in training data set
train = pd.DataFrame(pd.read_json("Two-Sigma_Rental-Listings/train.json"))
test = pd.read_json("Two-Sigma_Rental-Listings/test.json")

### Data summary
train.info()
train.describe()

### Class distribution for variable of interest. Significant class imbalance for high interest group. 
train.interest_level.value_counts() / train.interest_level.count()
# low = 0.695
# medium = 0.228
# high = 0.0778

### People will likely be more interested in listings that offer a good deal. 
# Median price for listing based on the number of bathrooms.
price_pivot = pd.pivot_table(train, index=['bedrooms'], 
                values=['price'], aggfunc=[np.median])
price_pivot.columns = ['medprice']

# Calculate the difference in between the listing price and the median price for that number of bedrooms.
beds_value = list()
for i in range(0, train.shape[0]):
        beds = train.iloc[i]['bedrooms']
        foo = train.iloc[i]['price'] - price_pivot.iloc[beds]['medprice']
        beds_value.append(foo)

# Add feature to data frame
train['beds_value'] = beds_value
# Remove outlier price points (listings greater thatn $13000) which will skew analysis.
priceDF = train[train.price < np.percentile(train.price, 99)]

# It appears price value has some predict power, but 
foo = [priceDF.beds_value[priceDF.interest_level == 'low'], 
        priceDF.beds_value[priceDF.interest_level == 'medium'],
        priceDF.beds_value[priceDF.interest_level == 'high']]

bp = plt.boxplot(foo, notch=0, sym='+', vert=1, whis=1.5)

pd.pivot_table(priceDF, index=['interest_level'],
        values=['beds_value'], aggfunc=[np.mean, np.std])


