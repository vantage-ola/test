# step 1
import pandas as pd

file_path = "./dataset/dataset.csv"

df = pd.read_csv(file_path)

# understand the structure
df.info(), df.head()

# step 2
import numpy as np

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Handle missing values
df_cleaned = df_cleaned.dropna(subset=['Description'])  # Remove rows where Description is missing

# Convert data types
df_cleaned['Quantity'] = pd.to_numeric(df_cleaned['Quantity'], errors='coerce')  # Convert to integer
df_cleaned['UnitPrice'] = pd.to_numeric(df_cleaned['UnitPrice'], errors='coerce')  # Convert to float
df_cleaned['CustomerID'] = pd.to_numeric(df_cleaned['CustomerID'], errors='coerce')  # Convert to float

# Convert InvoiceDate to datetime
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'], errors='coerce')

# Standardize Description (lowercase, remove special characters)
df_cleaned['Description'] = df_cleaned['Description'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)

# Clean Country column (remove unusual characters)
df_cleaned['Country'] = df_cleaned['Country'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.strip()

# Clean StockCode column (remove special characters)
df_cleaned['StockCode'] = df_cleaned['StockCode'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

# Drop rows with missing critical numeric values after conversion
df_cleaned = df_cleaned.dropna(subset=['Quantity', 'UnitPrice', 'CustomerID'])

# Save the cleaned dataset for further processing
cleaned_file_path = "./dataset/cleaned_dataset.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

# Display summary of cleaned dataset
df_cleaned.info(), df_cleaned.head()


# Further clean Country column by removing unintended prefixes like "XxY"
df_cleaned['Country'] = df_cleaned['Country'].str.replace(r'XxY', '', regex=True).str.strip()

# Remove non-standard characters from InvoiceNo
df_cleaned['InvoiceNo'] = df_cleaned['InvoiceNo'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

# Save the final cleaned dataset
final_cleaned_file_path = "./dataset/cleanded_dataset.csv"
df_cleaned.to_csv(final_cleaned_file_path, index=False)

# Display final sample
df_cleaned.head()
