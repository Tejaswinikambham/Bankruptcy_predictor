import joblib
import pandas as pd
def run_batch_prediction(file_path):
    model = joblib.load('model.pkl')
    data = pd.read_csv(file_path)
    
    predictions = model.predict(data)
    
    data['Prediction_Score'] = predictions
    
    data['Status'] = data['Prediction_Score'].apply(lambda x: 'Safe' if x < 0.5 else 'Unsafe')
    
    return data