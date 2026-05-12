import pandas as pd
import requests
import ast
import numpy as np

# Load the cleaned data
df = pd.read_csv('cleaned_matches.csv')

# Fetch hero data from OpenDota API
hero_url = "https://api.opendota.com/api/heroes"
response = requests.get(hero_url)
hero_data = response.json()

# Get all hero IDs from the dataset
all_hero_ids = set()
for team_list in df['radiant_team'].apply(ast.literal_eval):
    all_hero_ids.update(team_list)
for team_list in df['dire_team'].apply(ast.literal_eval):
    all_hero_ids.update(team_list)
all_hero_ids = sorted([h for h in all_hero_ids if h != 0])  # Remove placeholder 0

print(f"Found {len(all_hero_ids)} unique heroes in dataset")

# Create hero appearance features for Radiant team
# hero_X_radiant = 1 if hero X is in Radiant team, 0 otherwise
for hero_id in all_hero_ids:
    df[f'hero_{hero_id}_radiant'] = df['radiant_team'].apply(
        lambda x: 1 if hero_id in ast.literal_eval(x) else 0
    )

# Create target: radiant_win (1 = Radiant won, 0 = Dire won)
df['radiant_win'] = df['radiant_win'].astype(int)

# Calculate individual hero win rates when picked by Radiant
hero_stats = []
for hero_id in all_hero_ids:
    col = f'hero_{hero_id}_radiant'
    picked_count = df[col].sum()
    if picked_count > 0:
        win_rate = df[df[col] == 1]['radiant_win'].mean()
    else:
        win_rate = 0.5  # Default for never picked
    hero_stats.append({
        'hero_id': hero_id,
        'picks': picked_count,
        'wins': df[df[col] == 1]['radiant_win'].sum() if picked_count > 0 else 0,
        'win_rate': win_rate
    })

hero_stats_df = pd.DataFrame(hero_stats).sort_values('picks', ascending=False)
print("\nTop 15 Most Picked Heroes:")
print(hero_stats_df.head(15).to_string(index=False))

print("\nTop 10 Highest Win Rate Heroes (min 3 picks):")
high_win = hero_stats_df[hero_stats_df['picks'] >= 3].sort_values('win_rate', ascending=False)
print(high_win.head(10).to_string(index=False))

# Extract roles and attributes for analysis (keeping for reference)
hero_roles = {hero['id']: hero['roles'] for hero in hero_data}
hero_attributes = {hero['id']: hero['primary_attr'] for hero in hero_data}

# Create role/attribute features as additional context
def extract_roles(team):
    roles = []
    for hero_id in team:
        roles.extend(hero_roles.get(hero_id, []))
    return roles

def extract_attributes(team):
    return [hero_attributes.get(hero_id, 'unknown') for hero_id in team]

df['radiant_heroes'] = df['radiant_team'].apply(ast.literal_eval)
df['dire_heroes'] = df['dire_team'].apply(ast.literal_eval)
df['radiant_roles'] = df['radiant_heroes'].apply(extract_roles)
df['dire_roles'] = df['dire_heroes'].apply(extract_roles)
df['radiant_attributes'] = df['radiant_heroes'].apply(extract_attributes)
df['dire_attributes'] = df['dire_heroes'].apply(extract_attributes)

# Role features
all_roles = set()
for roles in df['radiant_roles'].tolist() + df['dire_roles'].tolist():
    all_roles.update(roles)

for role in sorted(all_roles):
    df[f'role_{role}_radiant'] = df['radiant_roles'].apply(lambda x: x.count(role))
    df[f'role_{role}_dire'] = df['dire_roles'].apply(lambda x: x.count(role))

# Attribute features (count)
all_attrs = ['str', 'agi', 'int', 'unknown']
for attr in all_attrs:
    df[f'attr_{attr}_radiant'] = df['radiant_attributes'].apply(lambda x: x.count(attr))
    df[f'attr_{attr}_dire'] = df['dire_attributes'].apply(lambda x: x.count(attr))

# Keep original features
keep_cols = ['radiant_win', 'start_time', 'duration', 'lobby_type', 'game_mode', 
             'avg_rank_tier', 'num_rank_tier', 'cluster']

# Get hero columns (these are our main features for hero analysis)
hero_cols = [col for col in df.columns if col.startswith('hero_')]
role_cols = [col for col in df.columns if col.startswith('role_')]
attr_cols = [col for col in df.columns if col.startswith('attr_')]

# Select final features
final_cols = keep_cols + hero_cols + role_cols + attr_cols
df_final = df[final_cols].copy()

# Save processed data
df_final.to_csv('processed_matches.csv', index=False)

# Also save hero statistics for reference
hero_stats_df.to_csv('hero_statistics.csv', index=False)

print(f"\nProcessed {len(df)} matches with {len(hero_cols)} hero features")
print("Features saved to processed_matches.csv")
print("Hero stats saved to hero_statistics.csv")
