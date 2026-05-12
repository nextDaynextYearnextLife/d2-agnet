import requests
import pandas as pd

# OpenDota API endpoint for match data
url = "https://api.opendota.com/api/publicMatches"

# Fetch data from the API
response = requests.get(url)
data = response.json()

# Print response status code and content for debugging
print(f"Response Status Code: {response.status_code}")
print(f"Response Content: {response.content}")

# Check if data is a list of dictionaries
if isinstance(data, list):
    # Convert to DataFrame
    df = pd.DataFrame(data)
else:
    # Handle case where data is a single dictionary
    df = pd.DataFrame([data])

# Save to CSV file
df.to_csv('matches.csv', index=False)

print("Data fetched and saved to matches.csv")
