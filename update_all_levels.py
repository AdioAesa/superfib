import os
import re
import glob
from datetime import datetime

def main():
    # Get the latest repeated_values file
    files = glob.glob("repeated_values_*.txt")
    if not files:
        print("No repeated_values files found")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Found latest file: {latest_file}")
    
    # Extract price levels from the file
    with open(latest_file, 'r') as f:
        content = f.read()
    
    # Extract levels and their frequencies
    price_levels = []
    frequency_labels = []
    
    # Find sections for different frequencies
    sections = re.findall(r'Values that appear (\d+) times:(.*?)(?=Values that appear|\Z)', content, re.DOTALL)
    
    # Process each frequency section - include ALL levels without limit
    for frequency, section in sections:
        # Extract levels from this section
        levels = re.findall(r'Level: (\d+\.\d+)', section)
        for level in levels:
            price_levels.append(float(level))
            frequency_labels.append(f"{frequency}X")
    
    # Read the current superfib.pine file
    with open("superfib.pine", 'r') as f:
        content = f.read()
    
    # Create string representations of the arrays
    prices_str = ", ".join(str(price) for price in price_levels)
    texts_str = ", ".join(f'"{label}"' for label in frequency_labels)
    
    # Create the replacement text
    comment = f"// Price levels from repeated_values analysis\n// Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n// All {len(price_levels)} levels from file: {latest_file}\n"
    prices_line = f"var prices = array.from({prices_str})  // Auto-generated price levels\n"
    texts_line = f"var texts = array.from({texts_str})  // Frequency labels"
    
    # Replace the price levels section
    pattern = r"// Price levels from repeated_values analysis.*?\n// Last updated:.*?\n// \d+ levels from.*?\nvar prices = array\.from\(.*?\).*?\nvar texts = array\.from\(.*?\).*?"
    replacement = f"{comment}{prices_line}{texts_line}"
    
    # Try to replace using the pattern
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # If the pattern didn't match, try a simpler approach
    if new_content == content:
        # Find where the arrays are defined
        price_pattern = r"var prices = array\.from\(.*?\).*?\n"
        text_pattern = r"var texts = array\.from\(.*?\).*?\n"
        
        # Replace the array definitions with the new ones
        new_content = re.sub(price_pattern, prices_line, content)
        new_content = re.sub(text_pattern, texts_line + "\n", new_content)
        
        # Also update the comment
        comment_pattern = r"// Price levels from repeated_values analysis.*?\n// Last updated:.*?\n// \d+ levels from.*?\n"
        if re.search(comment_pattern, new_content):
            new_content = re.sub(comment_pattern, comment, new_content)
        else:
            # If we can't find the comment, just insert it before the prices array
            new_content = re.sub(r"var prices = array\.from", comment + "var prices = array.from", new_content)
    
    # Write the updated file
    with open("superfib.pine", 'w') as f:
        f.write(new_content)
    
    print(f"Updated superfib.pine with {len(price_levels)} price levels")
    print(f"Source file: {latest_file}")

if __name__ == "__main__":
    main()
