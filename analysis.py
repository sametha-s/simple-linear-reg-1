import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# read data 
df = pd.read_csv("tvmarketing.csv")

# Display first 5 rows
df.head()

# Display last 5 rows
df.tail()

# Check shape of df
df.shape # (200, 2) -> 200 rows, 2 columns (TV and Sales)

# Visualise data using plot
df.plot(x = "TV", y = "Sales", kind = "scatter")

# Preparing X and y
X = df[["TV"]] # sklearn requires X to be a 2D array with shape (n_samples, n_features)
y = df["Sales"]

# Split data into Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 500)
# X_train = X_train[:, np.newaxis] # newaxis: alias for None; insert a new dimension of size 1 here

print(X_train.shape) # (140, 1)
print(y_train.shape) # (140, )
print(X_test.shape) # (60, 1)
print(y_test.shape) # (60, )

# Perform Linear Regression
lr = LinearRegression()


# fit the model
model = lr.fit(X_train, y_train)

# Print the intercept and coefficients
print(model.intercept_) # 6.75755
print(model.coef_) # 0.0506

# Predictions 
y_pred = model.predict(X_test)
type(y_pred)

# Actual vs Predicted
c = [i for i in range(1,61,1)]         # generating index 
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)   

# Error terms
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label

# Metrics
mse = mean_squared_error(y_test, y_pred)
print("MSE", mse)
# 12.92
# model is not able to match 12.92% of the data

r_squared = r2_score(y_test, y_pred)
print("R-Squared", r_squared) 
# 0.4829
# approx 48.29% of the variation is explained by the model

plt.figure()
plt.scatter(y_test,y_pred,c='blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.grid()

# Summary table
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)   # adds the intercept column
ols = sm.OLS(y_train, X_train_sm).fit()

print(ols.summary())
