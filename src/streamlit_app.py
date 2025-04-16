import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("models/billvision_model.pkl", "rb"))

# Load monthly dataset
monthly_df = pd.read_csv("data/monthly_consumption.csv", parse_dates=['Datetime'])
monthly_df.set_index('Datetime', inplace=True)

# Streamlit Page Setup
st.set_page_config(page_title="BillVision", page_icon="⚡", layout="centered")

# Theme toggle
theme = st.toggle("🌗 Toggle Dark Mode")

# Header
st.markdown(
    f"""
    <div style="text-align: center;">
        <h1 style="color:{'#C71585' if not theme else '#FF69B4'};">⚡ BillVision</h1>
        <h4 style="color:gray;">Smart Electricity Bill Predictor</h4>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")

# Input
st.subheader("🔢 Enter Monthly Usage")
kwh = st.number_input("Monthly Consumption (kWh)", min_value=0.0, step=1.0)

if st.button("🔮 Predict Bill"):
    prediction = model.predict(np.array([[kwh]]))
    amount = round(prediction[0], 2)

    st.success(f"📋 Estimated Bill: ₹{amount:.2f}")

    # Feedback
    if amount > 6000:
        st.warning("⚠️ High usage! Consider appliance audits or energy-saving devices.")
    elif amount < 1500:
        st.info("✅ Low consumption. Great job!")
    else:
        st.info("📉 Moderate usage. You’re doing okay!")

    st.markdown("---")

# Charts
st.subheader("📈 Electricity Usage Trends")

# Energy consumption chart
fig1, ax1 = plt.subplots()
monthly_df['Monthly_KWh'].plot(ax=ax1, color='royalblue' if not theme else 'skyblue')
ax1.set_title("Monthly Energy Consumption (kWh)")
ax1.set_ylabel("kWh")
st.pyplot(fig1)

# Bill trend chart
fig2, ax2 = plt.subplots()
monthly_df['Bill'].plot(ax=ax2, color='crimson' if not theme else 'tomato')
ax2.set_title("Estimated Monthly Bill (₹)")
ax2.set_ylabel("₹")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("<center style='color:gray;'>Made with ❤️ by BillVision</center>", unsafe_allow_html=True)
