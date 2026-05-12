import requests
import pandas as pd
import time
import os

url = "https://api.opendota.com/api/publicMatches"

def fetch_valid_matches(max_count=10000, batch_size=100, save_path='matches.csv'):
    """
    Fetch public matches from OpenDota API with valid hero data.
    Filters out matches where radiant_team = [0,0,0,0,0]
    """
    all_matches = []
    last_match_id = None
    skipped = 0
    
    print(f"Fetching up to {max_count} VALID matches from OpenDota API...")
    print("Filtering out matches with no hero data...")
    print("-" * 50)
    
    while len(all_matches) < max_count:
        params = {'limit': batch_size}
        if last_match_id:
            params['less_than_match_id'] = last_match_id
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            matches = response.json()
            
            if not matches:
                print(f"No more matches available")
                break
            
            # Filter valid matches (has hero data)
            for match in matches:
                radiant_team = match.get('radiant_team', [])
                # Check if team has at least 3 non-zero heroes
                if radiant_team and sum(1 for h in radiant_team if h != 0) >= 3:
                    all_matches.append(match)
                else:
                    skipped += 1
            
            last_match_id = matches[-1]['match_id']
            
            print(f"Valid: {len(all_matches):,} | Skipped: {skipped:,} | Total fetched: {len(all_matches)+skipped:,}")
            
            # Rate limiting
            time.sleep(1.2)
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if len(all_matches) > 0:
                break
            else:
                raise
    
    # Convert to DataFrame
    df = pd.DataFrame(all_matches)
    
    # Select columns
    cols = ['match_id', 'match_seq_num', 'radiant_win', 'start_time', 
            'duration', 'lobby_type', 'game_mode', 'avg_rank_tier', 
            'num_rank_tier', 'cluster', 'radiant_team', 'dire_team']
    df = df[[c for c in cols if c in df.columns]]
    
    # Save
    df.to_csv(save_path, index=False)
    
    print("-" * 50)
    print(f"Total valid matches: {len(df):,}")
    print(f"Skipped (no hero data): {skipped:,}")
    print(f"Saved to: {save_path}")
    
    return df

if __name__ == "__main__":
    df = fetch_valid_matches(max_count=10000)
