# Linear Regression is used to model linear relationship between dependent and independent variable.
# Assumptions of linear variable-
# 1>Linear Dependency 2>.Homosedaciticity 3>.Normal distribution of error terms 4>.Coorelation between error terms(meaning 
# we can derive next error using previous error term)5>Multicollinearity
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv('Transformed_Housing_Data2.csv')

# 1>.NOW WE WILL SCALE OUR DATA
scaler = StandardScaler()
Y = data['Sale_Price']
X = scaler.fit_transform(data.drop(columns = ['Sale_Price']))
X = pd.DataFrame(data = X, columns = data.drop(columns = ['Sale_Price']).columns)
X.head()
# 2>. We will check collinearity and remove it
# By using corr() funcation we can it but will have larger dataset. So we'll define our own corr() funcation
# in which if corr>0.5 then value will be displayed
k = X.corr()
z = [[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j] >abs(0.5))&(i!=j)]
z, len(z)
# 3>Now we will find VIF(variance influence factor)
vif_data = X
## Calculating VIF for every column
VIF = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index = vif_data.columns)
print(VIF[VIF == VIF.max()].index[0])
# now we will remove multicollinearity
def MC_remover(data):
  vif = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index = data.columns)
  if vif.max() > 5:
    print(vif[vif == vif.max()].index[0],'has been removed')
    data = data.drop(columns = [vif[vif == vif.max()].index[0]])
    return data
  else:
    print('No Multicollinearity present anymore')
    return data
# as we know that 7 dataset have multicollinearity greater than 5. So in order to remove this we will run for loop 7 times.
for i in range(7):
  vif_data = MC_remover(vif_data)
print(vif_data.head())
# checking VIF Value
VIF = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index = vif_data.columns)
print(VIF, len(vif_data.columns))
# Now We Will Train Our Dataset
X=vif_data
Y=data['Sale_Price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)
# Now apply linearRegression
lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)
# coef_ is used to find m1,m2,m3...
lr.coef_
predictions = lr.predict(x_test)
print(predictions)
# calculate R2 value
print(lr.score(x_test, y_test))

# Now we will look at the assumption of linear regression
# Residuals
residuals=predictions-y_test
residuals_table=pd.DataFrame({'residuals':residuals,
                            'predictions':predictions})
residuals_table.sort_values(by='predictions')
z=[i for i in range(int(residuals_table['predictions'].max()))]
k=[0 for i in range(int(residuals_table['predictions'].max()))]
plt.scatter(residuals_table['predictions'],residuals_table['residuals'],color="red")
plt.plot(z,k,color="green")
plt.xlabel('fitted points (ordered by predictions)')
plt.ylabel('residuals')
plt.title('residual plot')
plt.show()
# Now we will see disributions of error using histogram
plt.hist(residuals,color='green',bins=200)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.title('Error distribution curve')
plt.show()
# model coefficient
coefficients_table = pd.DataFrame({'column': x_train.columns,
                                  'coefficients': lr.coef_})
coefficient_table = coefficients_table.sort_values(by = 'coefficients')
x = coefficient_table['column']
y = coefficient_table['coefficients']
plt.barh( x, y)
plt.xlabel( "Coefficients")
plt.ylabel('Variables')
plt.title('Normalized Coefficient plot')
plt.show()
