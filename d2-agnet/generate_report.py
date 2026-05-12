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

# Calculate stats
total_matches = len(df)
radiant_win_rate = df['radiant_win'].mean()

# Fetch hero images
hero_url = "https://api.opendota.com/api/heroes"
hero_response = requests.get(hero_url)
hero_data = hero_response.json()

# Create visualization directory
os.makedirs('visualization', exist_ok=True)

# Generate feature bars HTML
feature_bars = ""
for _, row in feature_importance_df.iterrows():
    feature_name = row['Feature'].replace('_', ' ').title()
    importance = row['Importance'] * 100  # Convert to percentage
    feature_bars += f"""
        <div class="feature-item">
            <div class="feature-label">
                <span class="feature-name">{feature_name}</span>
                <span class="feature-value">{row['Importance']:.2%}</span>
            </div>
            <div class="feature-bar-bg">
                <div class="feature-bar" style="width: {importance:.1f}%"></div>
            </div>
        </div>
    """

# Generate hero cards
hero_cards = ""
for hero in hero_data[:24]:  # Show 24 heroes in grid
    hero_id = str(hero['id'])
    name = hero['name'].replace('npc_dota_hero_', '')
    local_img_path = f"hero_images/{hero_id}_{name}.png"
    if os.path.exists(f'visualization/{local_img_path}'):
        image_src = local_img_path
    else:
        image_src = f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{name}_lg.png"
    hero_name = hero['localized_name']
    # Get hero attributes for color coding
    primary_attr = hero.get('primary_attr', '')
    attr_color = {'agi': '#4CAF50', 'str': '#F44336', 'int': '#2196F3'}.get(primary_attr, '#9C27B0')
    attr_icon = {'agi': 'A', 'str': 'S', 'int': 'I'}.get(primary_attr, '?')
    
    hero_cards += f"""
        <div class="hero-card">
            <div class="hero-image-wrapper">
                <img src="{image_src}" alt="{hero_name}" onerror="this.src='https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{name}_lg.png'" />
                <span class="attr-badge" style="background: {attr_color}">{attr_icon}</span>
            </div>
            <span class="hero-name">{hero_name}</span>
        </div>
    """

# Full HTML template
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dota 2 Match Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 50px;
        }}
        
        .header h1 {{
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #ffd700, #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 3px;
        }}
        
        .header .subtitle {{
            color: #888;
            font-size: 1.1rem;
        }}
        
        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 50px;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 215, 0, 0.3);
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffd700;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #888;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Section */
        .section {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 40px;
            backdrop-filter: blur(5px);
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255, 215, 0, 0.3);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: linear-gradient(180deg, #ffd700, #ff8c00);
            border-radius: 2px;
        }}
        
        /* Feature Bars */
        .features-list {{
            display: flex;
            flex-direction: column;
            gap: 18px;
        }}
        
        .feature-item {{
            background: rgba(0, 0, 0, 0.2);
            padding: 15px 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .feature-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.95rem;
        }}
        
        .feature-name {{
            color: #ccc;
        }}
        
        .feature-value {{
            color: #ffd700;
            font-weight: 600;
        }}
        
        .feature-bar-bg {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .feature-bar {{
            height: 100%;
            background: linear-gradient(90deg, #ff8c00, #ffd700);
            border-radius: 4px;
            transition: width 1s ease;
        }}
        
        /* Hero Grid */
        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 20px;
        }}
        
        .hero-card {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        .hero-card:hover {{
            transform: scale(1.05);
            border-color: rgba(255, 215, 0, 0.5);
            box-shadow: 0 5px 20px rgba(255, 215, 0, 0.2);
        }}
        
        .hero-image-wrapper {{
            position: relative;
            width: 80px;
            height: 45px;
            margin: 0 auto 10px;
        }}
        
        .hero-image-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
        }}
        
        .attr-badge {{
            position: absolute;
            bottom: -5px;
            right: -5px;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
            color: #fff;
            border: 2px solid #1a1a2e;
        }}
        
        .hero-name {{
            font-size: 0.85rem;
            color: #ddd;
            font-weight: 500;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding-top: 30px;
            color: #555;
            font-size: 0.85rem;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            .hero-grid {{
                grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
                gap: 12px;
            }}
            .section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Dota 2 Match Analysis</h1>
            <p class="subtitle">Machine Learning Win Rate Prediction Report</p>
        </div>
        
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_matches}</div>
                <div class="stat-label">Total Matches</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{radiant_win_rate:.1%}</div>
                <div class="stat-label">Radiant Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(hero_data)}</div>
                <div class="stat-label">Heroes Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{model.n_estimators}</div>
                <div class="stat-label">Trees in Forest</div>
            </div>
        </div>
        
        <!-- Feature Importance -->
        <div class="section">
            <h2 class="section-title">Top 10 Feature Importance</h2>
            <div class="features-list">
                {feature_bars}
            </div>
        </div>
        
        <!-- Hero Gallery -->
        <div class="section">
            <h2 class="section-title">Hero Gallery</h2>
            <div class="hero-grid">
                {hero_cards}
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated by D2-agnet ML Model</p>
        </div>
    </div>
</body>
</html>"""

# Write final HTML
with open('visualization/report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Visualization generated at visualization/report.html")
