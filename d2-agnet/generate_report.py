import pandas as pd
import requests
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load processed data and model
df = pd.read_csv('processed_matches.csv')
model = joblib.load('random_forest_model.pkl')

# Get feature importances
importances = model.feature_importances_
features = df.columns[2:]  # Exclude match_id and radiant_win
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# Fetch hero images
hero_url = "https://api.opendota.com/api/heroes"
hero_response = requests.get(hero_url)
hero_data = hero_response.json()

# Create visualization directory
os.makedirs('visualization', exist_ok=True)

# Generate HTML report
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Dota2 Hero Win Rate Analysis</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        h1 { color: #333; }
        .hero-card { display: inline-block; margin: 10px; text-align: center; }
        img { width: 100px; height: 100px; }
    </style>
</head>
<body>
    <h1>Top 10 Important Features</h1>
    <table border="1">
        <tr><th>Feature</th><th>Importance</th></tr>
        {features_table}
    </table>

    <h1>Hero Win Rate Analysis</h1>
    {hero_cards}
</body>
</html>
"""

# Generate feature table
features_table = ""
for _, row in feature_importance_df.iterrows():
    features_table += f"<tr><td>{row['Feature']}</td><td>{row['Importance']:.4f}</td></tr>"

# Generate hero cards
hero_cards = ""
for hero in hero_data:
    hero_id = str(hero['id'])
    image_url = hero.get('img', '')
    if not image_url:
        continue
    win_rate = df[df[hero_id+'_radiant'].eq(1)]['radiant_win_rate'].mean()
    hero_cards += f"""
        <div class="hero-card">
            <img src="https://api.opendota.com{image_url}" alt="{hero['displayname']}" />
            <p>{hero['displayname']}</p>
            <p>Win Rate: {win_rate:.2%}</p>
        </div>
    """

# Escape curly braces in CSS
css_content = """
<style>
    body {{ font-family: Arial; padding: 20px; }}
    h1 {{ color: #333; }}
    .hero-card {{ display: inline-block; margin: 10px; text-align: center; }}
    img {{ width: 100px; height: 100px; }}
</style>
"""

# Generate HTML report
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dota2 Hero Win Rate Analysis</title>
    {css_content}
</head>
<body>
    <h1>Top 10 Important Features</h1>
    <table border="1">
        <tr><th>Feature</th><th>Importance</th></tr>
        {features_table}
    </table>

    <h1>Hero Win Rate Analysis</h1>
    {hero_cards}
</body>
</html>
"""

# Write final HTML
with open('visualization/report.html', 'w') as f:
    f.write(html_content)

print("Visualization generated at visualization/report.html")
