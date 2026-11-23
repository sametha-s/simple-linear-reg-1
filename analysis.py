import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
X = df[["TV"]
y = df["Sales"]

# Split data into Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 500)

print(X_train.shape) # (140, )
print(X_test.shape) # (60, )
print(y_train.shape)
print(y_test.shape)

# sklearn requires X to be a 2D array with shape (n_samples, n_features)
X_train = X_train[:, ]