import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
from streamlit_autorefresh import st_autorefresh

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Auto-refresh every 2 seconds
st_autorefresh(interval=2000, limit=None, key="attendance_refresh")

st.title("Attendance Dashboard")
st.write(f"Date: {date}")
st.caption("Auto-refreshing every 2 seconds")

file_path = "Attendance/Attendance_" + date + ".csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    if not df.empty:
        st.dataframe(df)
    else:
        st.write("No attendance records yet.")
else:
    st.write("No attendance records yet for today.")
