import pandas as pd
import requests
import ast

# Load the cleaned data
df = pd.read_csv('cleaned_matches.csv')

# Fetch hero data from OpenDota API
hero_url = "https://api.opendota.com/api/heroes"
response = requests.get(hero_url)
hero_data = response.json()

# Create dictionaries to map hero IDs to roles and attributes
hero_roles = {}
hero_attributes = {}
for hero in hero_data:
    hero_id = hero['id']
    hero_roles[hero_id] = hero['roles']
    hero_attributes[hero_id] = hero['primary_attr']

# Function to extract roles for a team
def extract_roles(team):
    roles = []
    for hero_id in team:
        roles.extend(hero_roles.get(hero_id, []))
    return roles

# Function to extract primary attributes for a team
def extract_attributes(team):
    attributes = []
    for hero_id in team:
        attributes.append(hero_attributes.get(hero_id, 'unknown'))
    return attributes

# Extract hero IDs from radiant_team and dire_team
df['radiant_heroes'] = df['radiant_team'].apply(ast.literal_eval)
df['dire_heroes'] = df['dire_team'].apply(ast.literal_eval)

# Extract roles and attributes for each team
df['radiant_roles'] = df['radiant_heroes'].apply(extract_roles)
df['dire_roles'] = df['dire_heroes'].apply(extract_roles)
df['radiant_attributes'] = df['radiant_heroes'].apply(extract_attributes)
df['dire_attributes'] = df['dire_heroes'].apply(extract_attributes)

# Flatten roles and attributes to create binary features
all_roles = set()
for roles in df['radiant_roles'] + df['dire_roles']:
    all_roles.update(roles)

all_attributes = set()
for attributes in df['radiant_attributes'] + df['dire_attributes']:
    all_attributes.update(attributes)

all_roles = sorted(all_roles)
all_attributes = sorted(all_attributes)

# Map primary attributes to numerical values
attr_map = {'str': 0, 'agi': 1, 'int': 2, 'unknown': -1}

# Create binary features for roles and numerical features for attributes
for role in all_roles:
    df[f'role_{role}_radiant'] = df['radiant_roles'].apply(lambda x: x.count(role) / 5)  # Normalize by team size
    df[f'role_{role}_dire'] = df['dire_roles'].apply(lambda x: x.count(role) / 5)

for attr in all_attributes:
    df[f'attr_{attr}_radiant'] = df['radiant_attributes'].apply(lambda x: x.count(attr) / 5)
    df[f'attr_{attr}_dire'] = df['dire_attributes'].apply(lambda x: x.count(attr) / 5)

# Create a column for the win rate of the radiant team
df['radiant_win_rate'] = df['radiant_win'].astype(int)

# Drop the original columns
df.drop(columns=['radiant_team', 'dire_team', 'radiant_heroes', 'dire_heroes', 'radiant_roles', 'dire_roles', 'radiant_attributes', 'dire_attributes'], inplace=True)

# Save the processed data
df.to_csv('processed_matches.csv', index=False)

print("Feature engineering completed and saved to processed_matches.csv")
