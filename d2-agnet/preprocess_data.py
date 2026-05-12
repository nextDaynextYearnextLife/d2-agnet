import pandas as pd

# Load the data
df = pd.read_csv('matches.csv')

# Display the first few rows of the dataframe
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Display the shape of the dataframe after dropping missing values
print(f"Shape after dropping missing values: {df.shape}")

# Save the cleaned data
df.to_csv('cleaned_matches.csv', index=False)

print("Data preprocessed and saved to cleaned_matches.csv")
