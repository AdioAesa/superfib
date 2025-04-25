from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap5
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

app = Flask(__name__)
bootstrap = Bootstrap5(app)

def calculate_fibonacci_levels(ticker, start_date, end_date):
    try:
        # Download historical data with user-provided date range
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            return {"error": f"No data found for ticker '{ticker}' in the specified date range"}
        
        df_input = stock_data[['High', 'Low']].copy()
        df_input.reset_index(inplace=True)
        
        # Calculate Fibonacci levels
        df_final = pd.DataFrame()
        df_final['Date'] = df_input['Date']
        df_final['High'] = df_input['High']
        df_final['Low'] = df_input['Low']
        
        daily_range = df_input['High'] - df_input['Low']
        fib_ratios = np.round(np.arange(-3.0, 3.0 + 0.25, 0.25), 3)
        
        for ratio in fib_ratios:
            current_ratio = float(ratio)
            if current_ratio < 0:
                df_final[current_ratio] = df_input['Low'] + daily_range * current_ratio
            else:
                df_final[current_ratio] = df_input['High'] + daily_range * current_ratio

        # Find repeated values
        numeric_df = df_final.select_dtypes(include=[np.number])
        all_values = numeric_df.values.flatten()
        all_values = np.round(all_values, 5)
        
        unique_values, counts = np.unique(all_values, return_counts=True)
        repeated_mask = counts > 1
        repeated_values = unique_values[repeated_mask]
        repeated_counts = counts[repeated_mask]
          # First sort by value (descending)
        value_sort_idx = np.argsort(-repeated_values)
        repeated_values = repeated_values[value_sort_idx]
        repeated_counts = repeated_counts[value_sort_idx]
        
        # Group by count
        count_groups = {}
        for value, count in zip(repeated_values, repeated_counts):
            if count not in count_groups:
                count_groups[count] = []
            count_groups[count].append(value)
        
        # Create the final sorted and grouped output
        result = []
        for count in sorted(count_groups.keys(), reverse=True):
            values_in_group = sorted(count_groups[count], reverse=True)
            for value in values_in_group:
                result.append({"value": float(value), "count": int(count)})
        
        return {
            "success": True,
            "repeated_values": result
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker')
    start_date = request.form.get('startDate')
    end_date = request.form.get('endDate')
    
    if not all([ticker, start_date, end_date]):
        return jsonify({"error": "Please provide ticker symbol and date range"})
    
    result = calculate_fibonacci_levels(ticker, start_date, end_date)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)