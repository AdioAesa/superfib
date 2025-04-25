# SuperFib

A comprehensive tool for generating and analyzing Fibonacci price levels for financial markets.

## Overview

SuperFib is a collection of tools for traders that:

1. Fetches historical stock/futures data using Yahoo Finance API
2. Calculates Fibonacci price levels using custom ratios 
3. Identifies repeated price levels that may act as important support/resistance
4. Exports data to Excel with conditional formatting
5. Generates a TradingView Pine Script indicator to visualize these levels on charts

## Components

- **main.py**: Downloads historical market data and calculates Fibonacci levels
- **check_duplicates.py**: Analyzes Excel files to find repeated price levels
- **update_all_levels.py**: Updates Pine Script with the latest price levels
- **superfib.pine**: TradingView Pine Script indicator for displaying price levels
- **app.py**: Web interface for the application (Flask-based)

## How to Use

### Step 1: Generate Fibonacci Levels

Run the main script to download historical data and generate Fibonacci levels for a specific ticker:

```bash
python main.py
```

This will:
- Download 6 months of historical data for the configured ticker (default: TSLA)
- Calculate Fibonacci levels using custom ratios
- Export data to an Excel file with highlighted repeated levels

### Step 2: Analyze Repeated Levels

After generating the Excel file, run the check_duplicates script to analyze repeated price levels:

```bash
python check_duplicates.py
```

This will:
- Scan the Excel file for price levels that appear multiple times
- Group levels by frequency (2X, 3X, 4X appearances)
- Save the results to a text file

### Step 3: Update the TradingView Indicator

To update the Pine Script indicator with the latest price levels:

```bash
python update_all_levels.py
```

This will:
- Extract all price levels from the latest analysis
- Update the `superfib.pine` file with these levels

### Step 4: Use in TradingView

1. Copy the contents of the `superfib.pine` file
2. Paste into TradingView Pine Script Editor
3. Add to your chart

## Web Interface

A simple web interface is available by running:

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## Requirements

- Python 3.6+
- pandas
- numpy
- yfinance
- xlsxwriter
- Flask (for web interface)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
