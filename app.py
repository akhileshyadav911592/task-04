import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("sales_model.pkl")

st.title("Sales Forecasting System")

month = st.number_input(
    "Enter Future Month Number",
    min_value=1,
    value=25
)

if st.button("Predict Sales"):
    prediction = model.predict([[month]])[0]

    st.success(
        f"Predicted Sales for Month {month}: ₹{prediction:.2f}"
    )

# Load data
df = pd.read_csv("sales_data.csv")

# Historical sales chart
st.subheader("Historical Sales Trend")

fig, ax = plt.subplots()
ax.plot(df["Month"], df["Sales"], marker='o')
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Sales Trend")

st.pyplot(fig)

# Forecast chart
future_months = list(range(1, 37))
future_sales = model.predict(
    pd.DataFrame(future_months, columns=["Month"])
)

forecast_df = pd.DataFrame({
    "Month": future_months,
    "Forecast Sales": future_sales
})

st.subheader("Sales Forecast")

fig2, ax2 = plt.subplots()
ax2.plot(
    forecast_df["Month"],
    forecast_df["Forecast Sales"],
    marker='o'
)
ax2.set_xlabel("Month")
ax2.set_ylabel("Forecast Sales")
ax2.set_title("Future Sales Forecast")

st.pyplot(fig2)