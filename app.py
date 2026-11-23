import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.title("TV Marketing Simple Linear Regression")

df = pd.read_csv("tvmarketing.csv")
st.subheader("First few rows of the dataset")
st.dataframe(df.head())

X = df[["TV"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=500)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

st.subheader("Model Summary")
ols = smf.ols("Sales ~ TV", data=df).fit()
coef_table = pd.DataFrame({
    "Coefficient": ols.params,
    "Std Error": ols.bse,
    "t-value": ols.tvalues,
    "p-value": ols.pvalues,
})
coef_table = coef_table.join(ols.conf_int().rename(columns={0: "CI 2.5%", 1: "CI 97.5%"}))
st.dataframe(coef_table)

# PLOTS TABS -------------------------------------------------------------------
st.subheader("Plots")
tab1, tab2, tab3 = st.tabs(["Sales vs TV Scatterplot", "Actual vs Predicted Plot", "Error Terms"])

with tab1:
    fig, ax = plt.subplots()
    ax.scatter(df["TV"], df["Sales"], color="blue")
    ax.set_xlabel("TV")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

with tab2:
  # Actual vs Predicted
  c = range(1, len(y_test) + 1)   # index that always matches the test set size
  fig, ax = plt.subplots()
  ax.plot(c, y_test, color="blue", linewidth=2, linestyle="-")
  ax.plot(c, y_pred, color="red", linewidth=2, linestyle="-")
  ax.set_title("Actual and Predicted", fontsize=20)
  ax.set_xlabel("Index", fontsize=18)
  ax.set_ylabel("Sales", fontsize=16)
  st.pyplot(fig)

with tab3:
  c = range(1, len(y_test) + 1)
  fig, ax = plt.subplots()
  ax.plot(c, y_test-y_pred, color="blue", linewidth=2, linestyle="-")
  ax.set_title("Error Terms")
  ax.set_xlabel('Index', fontsize=18)                      # X-label
  ax.set_ylabel('ytest-ypred', fontsize=16)
  st.pyplot(fig)    

# ----------------------------------------------------------------------------