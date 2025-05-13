import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter

# --- 1. Configuration ---
ticker = "NQ=F" # Example: "AAPL" for Apple, "ETH-USD" for Ethereum
# Note: Ensure the ticker is valid for Yahoo Finance
# output_filename = f'{ticker}_fibonacci_levels.xlsx' # Will be set based on calculation_type
num_decimals = 4
calculation_type = "monthly" # Options: "daily", "weekly", or "monthly". Change this to switch.

if calculation_type == "daily":
    output_filename = f'{ticker}_fibonacci_levels_daily.xlsx'
    sheet_name_excel = 'Fibonacci Levels Daily'
elif calculation_type == "weekly":
    output_filename = f'{ticker}_fibonacci_levels_weekly.xlsx'
    sheet_name_excel = 'Fibonacci Levels Weekly'
elif calculation_type == "monthly":
    output_filename = f'{ticker}_fibonacci_levels_monthly.xlsx'
    sheet_name_excel = 'Fibonacci Levels Monthly'
else:
    print("Error: Invalid calculation_type. Choose 'daily', 'weekly', or 'monthly'.")
    exit()

# --- 2. Define Date Range (Last 3 Months) ---
end_date = datetime.today()
start_date = end_date - relativedelta(months=60)  # Adjusted to 6 months for more data
# Ensure the start date is a weekday (Monday to Friday)
if start_date.weekday() >= 5:
    start_date += timedelta(days=(7 - start_date.weekday()))
elif start_date.weekday() == 6:
    start_date += timedelta(days=1)
print(f"Fetching data for ticker: {ticker}")
print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Calculation type: {calculation_type}")

# --- 3. Fetch Live Stock Data ---
try:
    # Download historical stock data from Yahoo Finance
    stock_data_downloaded = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if stock_data_downloaded.empty:
        print(f"No data found for stock '{ticker}' in the specified date range.")
        exit()
    else:
        print("Data fetched successfully.")
        # Data from yf.download has 'Date' as index. Reset index to make 'Date' a column.
        processed_data = stock_data_downloaded.reset_index()
        
        if calculation_type == "weekly" or calculation_type == "monthly":
            frequency_text = "weekly" if calculation_type == "weekly" else "monthly"
            print(f"Resampling data to {frequency_text} frequency...")
            processed_data['Date'] = pd.to_datetime(processed_data['Date'])
            
            # Ensure we have High and Low for aggregation
            high_col = 'High'
            low_col = 'Low'
            
            # Handle MultiIndex columns if present (yfinance sometimes returns these)
            if isinstance(processed_data.columns, pd.MultiIndex):
                print("MultiIndex columns detected - adjusting column access")
                # Find the correct column names for High and Low in a MultiIndex
                for col in processed_data.columns:
                    if isinstance(col, tuple) and col[0] == 'High':
                        high_col = col
                    if isinstance(col, tuple) and col[0] == 'Low':
                        low_col = col
            
            if high_col not in processed_data.columns or low_col not in processed_data.columns:
                raise ValueError(f"Downloaded data is missing 'High' or 'Low' columns for {frequency_text} resampling.")
            
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
            
            for time_period, group in period_grouped:
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
            
            print(f"{calculation_type.capitalize()} resampling complete.")
            print(f"{calculation_type.capitalize()} data sample:")
            print(df_input.head())
        else: # Daily
            # Select necessary columns. 'Date' is already a column from reset_index() above.
            df_input = processed_data[['Date', 'High', 'Low']].copy()
            print("Using daily data.")

except Exception as e:
    print(f"An error occurred while fetching stock data: {e}")
    print("Note: Make sure the stock symbol is valid for Yahoo Finance")
    exit()

# --- 4. Define NEW Fibonacci Ratios (negative: -6 to 0, positive: 0 to +6, excluding 0, 0.25, 0.75) ---
# Create arrays for negative and positive values
neg_ratios = np.round(np.arange(-6.0, 0, 0.25), num_decimals)
pos_ratios = np.round(np.arange(0.5, 6.0 + 0.25, 0.25), num_decimals)
# Combine and filter out unwanted values
fib_ratios = np.array([x for x in np.concatenate([neg_ratios, pos_ratios]) if x not in [0, 0.25, 0.75]])
print(f"\nUsing Fibonacci ratios: {fib_ratios.tolist()}")


# --- 5. Calculate Fibonacci Levels and Create Final DataFrame (Using Updated Logic) ---
df_final = pd.DataFrame()
df_final['Date'] = df_input['Date']
df_final['High'] = df_input['High']
df_final['Low'] = df_input['Low']

period_range = df_input['High'] - df_input['Low'] # Renamed from daily_range

# Store the names of the calculated fib columns for later use
fib_col_names = [float(ratio) for ratio in fib_ratios]

print("\nCalculating Fibonacci levels with updated logic...")
for ratio in fib_ratios:
    current_ratio = float(ratio)
    if current_ratio < 0:
        df_final[current_ratio] = df_input['Low'] + period_range * current_ratio
    elif current_ratio == 0.5:  # Special case for 0.5
        df_final[current_ratio] = df_input['Low'] + period_range * current_ratio
    else:
        df_final[current_ratio] = df_input['High'] + period_range * current_ratio
print("Calculations complete.")

# --- 6. Export to Excel with Conditional Formatting ---
print(f"\nExporting data to '{output_filename}' with highlighting...")

# Use ExcelWriter with xlsxwriter engine
with pd.ExcelWriter(output_filename, engine='xlsxwriter', date_format='yyyy-mm-dd') as writer:
    # Write the dataframe without default formatting (we'll apply custom formats)
    df_final.to_excel(writer, sheet_name=sheet_name_excel, index=False)

    # Get xlsxwriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets[sheet_name_excel]

    # Define formats
    # Number format string matching the desired decimal places
    num_format_str = f'0.{"0"*num_decimals}'

    # Default format for numbers
    default_format = workbook.add_format({'num_format': num_format_str})
    # Highlight formats
    yellow_format = workbook.add_format({'bg_color': '#FFFF00', 'num_format': num_format_str}) # Yellow for 2 repeats
    red_format = workbook.add_format({'bg_color': '#FFC7CE', 'num_format': num_format_str})    # Red for 3 repeats (light red fill)
    green_format = workbook.add_format({'bg_color': '#C6EFCE', 'num_format': num_format_str})   # Green for 4 repeats (light green fill)

    # Apply formatting row by row
    # Start from row 1 (row 0 is headers)
    for row_idx in range(len(df_final)):
        excel_row = row_idx + 1 # Excel rows are 1-based

        # Get the calculated fib values for the current row, rounded for comparison
        try:
             # Select only the columns that correspond to fib ratios
             row_fib_values = df_final.loc[row_idx, fib_col_names].round(num_decimals)
             # Count frequencies of values in this row
             value_counts = Counter(row_fib_values)
        except KeyError as e:
             print(f"Warning: Skipping row {excel_row} due to KeyError: {e}. Check column names.")
             continue # Skip to next row if there's an issue accessing columns


        # Iterate through the Fibonacci columns to apply formatting
        for ratio in fib_col_names:
            try:
                # Get the column index in the Excel sheet
                col_idx = df_final.columns.get_loc(ratio)
                # Get the original (unrounded) value to write
                value = df_final.loc[row_idx, ratio]
                # Get the rounded value for checking frequency
                rounded_value = round(value, num_decimals)

                # Determine the count and apply the appropriate format
                count = value_counts[rounded_value]

                cell_format = default_format # Start with default
                if count == 2:
                    cell_format = yellow_format
                elif count == 3:
                    cell_format = red_format
                elif count == 4:
                    cell_format = green_format
                # else: use default_format (already set)

                # Write the value with the chosen format
                # Check for NaN or infinite values which cause errors in xlsxwriter
                if pd.isna(value) or not np.isfinite(value):
                     worksheet.write(excel_row, col_idx, '', cell_format) # Write empty string for non-finite numbers
                else:
                     worksheet.write(excel_row, col_idx, value, cell_format)

            except KeyError:
                 # This might happen if a ratio column wasn't created correctly
                 print(f"Warning: Column {ratio} not found for formatting at row {excel_row}.")
                 continue # Skip this cell
            except TypeError as te:
                 print(f"Warning: TypeError writing cell ({excel_row}, {col_idx}) with value {value}. Error: {te}")
                 worksheet.write(excel_row, col_idx, str(value), default_format) # Try writing as string

    # Apply default format to non-Fibonacci columns (Date, High, Low) for consistency if needed
    # Example for Date column (adjust format as needed)
    date_col_idx = df_final.columns.get_loc('Date')
    date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'}) # Specific date format
    worksheet.set_column(date_col_idx, date_col_idx, width=12, cell_format=date_format)

    high_col_idx = df_final.columns.get_loc('High')
    low_col_idx = df_final.columns.get_loc('Low')
    worksheet.set_column(high_col_idx, high_col_idx, width=12, cell_format=default_format)
    worksheet.set_column(low_col_idx, low_col_idx, width=12, cell_format=default_format)

    # Adjust column widths for Fibonacci columns (optional)
    first_fib_col_idx = df_final.columns.get_loc(fib_col_names[0])
    last_fib_col_idx = df_final.columns.get_loc(fib_col_names[-1])
    worksheet.set_column(first_fib_col_idx, last_fib_col_idx, width=12) # Adjust width as needed


print(f"\nExcel file '{output_filename}' created successfully with highlighting.")
