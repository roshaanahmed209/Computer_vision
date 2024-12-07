import streamlit as st
import requests

# Streamlit UI
st.set_page_config(page_title="X-Ray DiagnosisBot", layout="centered")
st.title("X-Ray DiagnosisBot")
st.markdown("Upload the paths to the data directory and annotations file to generate a report.")

# Input fields
data_dir = st.text_input("Enter the Data Directory Path:")
anno_path = st.text_input("Enter the Annotations File Path:")

if st.button("Generate Report"):
    if not data_dir or not anno_path:
        st.error("Please provide both the data directory and annotations path.")
    else:
        # API call
        try:
            response = requests.post(
                "http://127.0.0.1:8000/evaluate",
                data={"data_dir": data_dir, "anno_path": anno_path},
            )
            if response.status_code == 200:
                report = response.json().get("report", "No report generated.")
                st.text_area("Generated Report", value=report, height=300)
            else:
                st.error(f"Failed to fetch the report. Error: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


