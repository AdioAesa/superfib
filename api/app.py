from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import sys

# Configure Flask to find templates in the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(parent_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

def calculate_fibonacci_levels(ticker, start_date, end_date, calculation_type="daily"):
    try:
        # Download historical data with user-provided date range
        stock_data_downloaded = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data_downloaded.empty:
            return {"error": f"No data found for ticker '{ticker}' in the specified date range"}
        
        processed_data = stock_data_downloaded.reset_index()
        
        if calculation_type == "weekly" or calculation_type == "monthly":
            # Handle resampling based on calculation type
            resample_text = "weekly" if calculation_type == "weekly" else "monthly"
            processed_data['Date'] = pd.to_datetime(processed_data['Date'])
            
            # Handle MultiIndex columns if present (yfinance sometimes returns these)
            high_col = 'High' 
            low_col = 'Low'
            
            if isinstance(processed_data.columns, pd.MultiIndex):
                # Find the correct column names for High and Low in a MultiIndex
                for col in processed_data.columns:
                    if isinstance(col, tuple) and col[0] == 'High':
                        high_col = col
                    if isinstance(col, tuple) and col[0] == 'Low':
                        low_col = col
            
            if high_col not in processed_data.columns or low_col not in processed_data.columns:
                return {"error": f"Downloaded data is missing 'High' or 'Low' columns for {resample_text} resampling."}
                
            # Create appropriate time period identifier for groupby operation
            if calculation_type == "weekly":
                # Year-weeknumber for weekly
                processed_data['TimePeriod'] = processed_data['Date'].dt.strftime('%Y-%U')
            else:
                # Year-month for monthly
                processed_data['TimePeriod'] = processed_data['Date'].dt.strftime('%Y-%m')
            
            # Group by time period and calculate high/low
            period_grouped = processed_data.groupby('TimePeriod')
            
            # Create period DataFrame from scratch with explicit columns
            period_dates = []
            period_highs = []
            period_lows = []
            
            for period_id, group in period_grouped:
                dates_sorted = sorted(group['Date'])
                
                if calculation_type == "weekly":
                    # For weekly: try to find Monday, otherwise use first day
                    reference_date = None
                    for date in dates_sorted:
                        if date.weekday() == 0:  # 0 = Monday
                            reference_date = date
                            break
                    
                    # If no Monday found, use the first date of the week
                    if reference_date is None and len(dates_sorted) > 0:
                        reference_date = dates_sorted[0]
                else:
                    # For monthly: use the first day of the month
                    reference_date = dates_sorted[0] if len(dates_sorted) > 0 else None
                
                if reference_date is not None:
                    # Extract the highest high and lowest low for the period
                    high_value = group[high_col].max()
                    low_value = group[low_col].min()
                    
                    # If the result is a Series (which can happen with MultiIndex), extract the float value
                    if isinstance(high_value, pd.Series):
                        high_value = high_value.iloc[0]  # Get the actual float value
                    if isinstance(low_value, pd.Series):
                        low_value = low_value.iloc[0]  # Get the actual float value
                    
                    period_dates.append(reference_date)
                    period_highs.append(high_value)
                    period_lows.append(low_value)
            
            # Create period DataFrame with explicit columns
            df_input = pd.DataFrame({
                'Date': period_dates,
                'High': period_highs,
                'Low': period_lows
            })
            
        else: # Daily
            df_input = processed_data[['Date', 'High', 'Low']].copy()
            
        # Calculate Fibonacci levels
        df_final = pd.DataFrame()
        df_final['Date'] = df_input['Date']
        df_final['High'] = df_input['High']
        df_final['Low'] = df_input['Low']
        
        period_range = df_input['High'] - df_input['Low'] # Renamed from daily_range
        
        # Define Fibonacci Ratios (identical to main.py)
        num_decimals = 4  # Match main.py precision
        neg_ratios = np.round(np.arange(-6.0, 0, 0.25), num_decimals)
        pos_ratios = np.round(np.arange(0.5, 6.0 + 0.25, 0.25), num_decimals)
        # Combine and filter out unwanted values
        fib_ratios = np.array([x for x in np.concatenate([neg_ratios, pos_ratios]) if x not in [0, 0.25, 0.75]])
        
        for ratio in fib_ratios:
            current_ratio = float(ratio)
            if current_ratio < 0:
                df_final[current_ratio] = df_input['Low'] + period_range * current_ratio
            else:
                df_final[current_ratio] = df_input['High'] + period_range * current_ratio
        
        # Find repeated values - using exact logic from check_duplicates.py
        
        # Get numeric columns only
        numeric_df = df_final.select_dtypes(include=[np.number])
        
        # Create a flat list of all values
        all_values = numeric_df.values.flatten()
        
        # Round values to 2 decimal places to avoid floating point issues
        all_values = np.round(all_values, 2)
        
        # Determine if dataset spans more than 3 months
        skip_2x_values = False
        if 'Date' in df_final.columns and pd.api.types.is_datetime64_any_dtype(df_final['Date']):
            date_range = df_final['Date'].max() - df_final['Date'].min()
            months = date_range.days / 30.44  # Average days per month
            # Skip 2X values only if: range > 3 months AND NOT weekly or monthly calculation
            skip_2x_values = months > 3 and calculation_type not in ["weekly", "monthly"]
        
        # Find unique values and their counts
        unique_values, counts = np.unique(all_values, return_counts=True)
        
        # Get repeated values (count > 1)
        if skip_2x_values:
            # Only include values that appear 3 or more times
            repeated_mask = counts > 2
        else:
            # Include all repeated values (appearing 2 or more times)
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
    calculation_type = request.form.get('calculationType', 'daily') # Get calculation_type, default to 'daily'
    
    if not all([ticker, start_date, end_date]):
        return jsonify({"error": "Please provide ticker symbol and date range"})
    
    result = calculate_fibonacci_levels(ticker, start_date, end_date, calculation_type)
    return jsonify(result)

# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

# Vercel requires this for serverless deployments
@app.route('/<path:path>')
def catch_all(path):
    return home()

if __name__ == '__main__':
    app.run(debug=True)
