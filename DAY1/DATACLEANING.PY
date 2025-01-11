import pandas as pd
import re
from google.colab import files

# Upload the CSV file
uploaded = files.upload()

# Specify your filename
filename = list(uploaded.keys())[0]

# Load the CSV file into a DataFrame
df = pd.read_csv(filename)

# Fill null values in the 'society' column
df['society'] = df['society'].fillna("Unknown Society")

# Fill null values in the 'balcony' column with 0 (default value)
df['balcony'] = df['balcony'].fillna(0).astype(int)  # Ensure values are integers

# Standardize the 'size' column to only integers
def standardize_size(size):
    if pd.isnull(size):
        return 0  # Replace null values with 0

    size = str(size).lower()

    # Extract the first integer in the size field
    match = re.search(r'(\d+)', size)
    if match:
        return int(match.group(1))  # Return the integer as a numeric value

    return 0  # Default if no number is found

# Apply the transformation to the 'size' column
df['size'] = df['size'].apply(standardize_size)

# Standardize the 'total square feet' column to integers
def standardize_sqft(sqft):
    if pd.isnull(sqft):
        return 0  # Replace null values with 0

    sqft = str(sqft).lower()

    # Extract the first integer in the sqft field
    match = re.search(r'(\d+)', sqft)
    if match:
        return int(match.group(1))  # Return the integer as a numeric value

    return 0  # Default if no number is found

# Apply the transformation to the 'total square feet' column
if 'total_sqft' in df.columns:
    df['total_sqft'] = df['total_sqft'].apply(standardize_sqft)

# Save the updated DataFrame to a new CSV
updated_filename = 'cleaned_file_with_integers.csv'
df.to_csv(updated_filename, index=False)

# Download the updated file
files.download(updated_filename)

print("File has been updated and downloaded.")
