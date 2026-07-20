import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np


st.set_page_config(page_title="Company Bankruptcy Predictor", layout="wide")

columns_sample=[' ROA(A) before interest and % after tax', ' Operating Gross Margin', ' Operating Profit Rate', ' Non-industry income and expenditure/revenue', ' Operating Expense Rate', ' Research and development expense rate', ' Cash flow rate', ' Interest-bearing debt interest rate', ' Tax rate (A)', ' Net Value Per Share (A)', ' Persistent EPS in the Last Four Seasons', ' Cash Flow Per Share', ' Revenue Per Share (Yuan ¥)', ' Realized Sales Gross Profit Growth Rate', ' Operating Profit Growth Rate', ' Regular Net Profit Growth Rate', ' Continuous Net Profit Growth Rate', ' Total Asset Growth Rate', ' Net Value Growth Rate', ' Total Asset Return Growth Rate Ratio', ' Cash Reinvestment %', ' Current Ratio', ' Quick Ratio', ' Interest Expense Ratio', ' Total debt/Total net worth', ' Debt ratio %', ' Long-term fund suitability ratio (A)', ' Borrowing dependency', ' Contingent liabilities/Net worth', ' Operating profit/Paid-in capital', ' Inventory and accounts receivable/Net value', ' Total Asset Turnover', ' Accounts Receivable Turnover', ' Average Collection Days', ' Inventory Turnover Rate (times)', ' Fixed Assets Turnover Frequency', ' Net Worth Turnover Rate (times)', ' Revenue per person', ' Operating profit per person', ' Allocation rate per person', ' Working Capital to Total Assets', ' Quick Assets/Total Assets', ' Current Assets/Total Assets', ' Cash/Total Assets', ' Quick Assets/Current Liability', ' Cash/Current Liability', ' Current Liability to Assets', ' Operating Funds to Liability', ' Inventory/Working Capital', ' Inventory/Current Liability', ' Current Liabilities/Liability', ' Working Capital/Equity', ' Current Liabilities/Equity', ' Long-term Liability to Current Assets', ' Retained Earnings to Total Assets', ' Total income/Total expense', ' Total expense/Assets', ' Current Asset Turnover Rate', ' Quick Asset Turnover Rate', ' Working capitcal Turnover Rate', ' Cash Turnover Rate', ' Fixed Assets to Assets', ' Equity to Long-term Liability', ' Cash Flow to Total Assets', ' Cash Flow to Liability', ' CFO to Assets', ' Cash Flow to Equity', ' Current Liability to Current Assets', ' Total assets to GNP price', ' No-credit Interval', " Net Income to Stockholder's Equity", ' Degree of Financial Leverage (DFL)', ' Interest Coverage Ratio (Interest expense to EBIT)', ' Equity to Liability']


def run_batch_prediction(file_path):
    model = joblib.load('model.pkl')
    
    data = pd.read_csv(file_path)
    
    predictions = model.predict(data)
    
    data['Prediction_Score'] = predictions

    data['Status'] = data['Prediction_Score'].apply(lambda x: 'Safe' if x < 0.5 else 'Unsafe')
    
    return data


st.title("📊 Company Bankruptcy Risk Dashboard")
st.write("Upload a CSV file containing company financial features to assess bankruptcy risk.")

def get_sample_template():
   
    columns = [f'feature_{i+1}' for i in range(74)]
    sample_data = pd.DataFrame(np.zeros((1, 74)), columns=columns_sample)
    return sample_data.to_csv(index=False).encode('utf-8')


st.subheader("Need help getting started?")
st.download_button(
    label="📥 Download Sample CSV Template",
    data=get_sample_template(),
    file_name='sample_template.csv',
    mime='text/csv',
)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="bankruptcy_uploader")
if uploaded_file is not None:
    uploaded_file.seek(0)
    
    try:
     
        result_df = run_batch_prediction(uploaded_file)
        
        st.success("Analysis complete!")
        
        
        col1, col2 = st.columns(2)
        safe_count = (result_df['Status'] == 'Safe').sum()
        unsafe_count = (result_df['Status'] == 'Unsafe').sum()
        
        col1.metric("Safe Companies", safe_count)
        col2.metric("Unsafe Companies", unsafe_count)
    
        fig = px.pie(result_df, names='Status', title="Company Safety Distribution", 
                     color='Status', color_discrete_map={'Safe':'green', 'Unsafe':'red'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Analysis Results")
        
        def color_status(val):
            color = 'background-color: #d4edda' if val == 'Safe' else 'background-color: #f8d7da'
            return color
        
        st.dataframe(result_df.style.map(color_status, subset=['Status']), use_container_width=True)        
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='analysis_results.csv',
            mime='text/csv',
        )
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")


