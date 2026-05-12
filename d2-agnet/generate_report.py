import pandas as pd
import requests
import os
import joblib

# Load processed data and model
df = pd.read_csv('processed_matches.csv')
model = joblib.load('random_forest_model.pkl')

# Load hero statistics
hero_stats = pd.read_csv('hero_statistics.csv')

# Fetch hero data
hero_url = "https://api.opendota.com/api/heroes"
hero_response = requests.get(hero_url)
hero_data = {h['id']: h for h in hero_response.json()}

# Calculate stats
total_matches = len(df)
radiant_win_rate = df['radiant_win'].mean()

# Get feature importances
importances = model.feature_importances_
feature_names = df.drop(columns=['radiant_win']).columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Hero features analysis
hero_cols = [c for c in feature_names if c.startswith('hero_')]
hero_importance = importance_df[importance_df['feature'].isin(hero_cols)]

# Add hero names to importance
def hero_id_from_col(col):
    return int(col.replace('hero_', '').replace('_radiant', ''))

hero_importance['hero_id'] = hero_importance['feature'].apply(hero_id_from_col)
hero_importance['hero_name'] = hero_importance['hero_id'].map(lambda x: hero_data.get(x, {}).get('localized_name', f'Hero {x}'))

# Add win rate to hero importance
hero_stats_dict = hero_stats.set_index('hero_id').to_dict()
hero_importance['pick_count'] = hero_importance['hero_id'].map(hero_stats_dict.get('picks', {}))
hero_importance['win_rate'] = hero_importance['hero_id'].map(hero_stats_dict.get('win_rate', {}))

# Create visualization directory
os.makedirs('visualization', exist_ok=True)

# Generate feature bars HTML
feature_bars = ""
for _, row in importance_df.head(12).iterrows():
    feature_name = row['feature'].replace('_', ' ').title()
    importance = row['importance'] * 100
    feature_bars += f"""
        <div class="feature-item">
            <div class="feature-label">
                <span class="feature-name">{feature_name}</span>
                <span class="feature-value">{row['importance']:.3f}</span>
            </div>
            <div class="feature-bar-bg">
                <div class="feature-bar" style="width: {importance:.1f}%"></div>
            </div>
        </div>
    """

# Generate top/bottom hero cards
def create_hero_card(hero_id, hero_name, win_rate, picks, importance, hero_data):
    name = hero_data.get('name', '').replace('npc_dota_hero_', '')
    local_img = f"hero_images/{hero_id}_{name}.png"
    if os.path.exists(f'visualization/{local_img}'):
        img_src = local_img
    else:
        img_src = f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{name}_lg.png"
    
    attr = hero_data.get('primary_attr', '')
    attr_colors = {'agi': '#4CAF50', 'str': '#F44336', 'int': '#2196F3'}
    attr_color = attr_colors.get(attr, '#9C27B0')
    
    return f"""
        <div class="hero-card">
            <div class="hero-image-wrapper">
                <img src="{img_src}" alt="{hero_name}" onerror="this.src='https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{name}_lg.png'" />
            </div>
            <div class="hero-info">
                <span class="hero-name">{hero_name}</span>
                <div class="hero-stats">
                    <span class="stat win-rate" style="color: {'#4CAF50' if win_rate >= 0.5 else '#F44336'}">{win_rate:.1%}</span>
                    <span class="stat picks">{picks} picks</span>
                </div>
            </div>
        </div>
    """

# Top 10 win rate heroes
top_heroes_html = ""
top_heroes = hero_importance[hero_importance['pick_count'] >= 10].nlargest(10, 'win_rate')
for _, row in top_heroes.iterrows():
    top_heroes_html += create_hero_card(
        row['hero_id'], row['hero_name'], row['win_rate'], 
        int(row['pick_count']), row['importance'], 
        hero_data.get(row['hero_id'], {})
    )

# Bottom 10 win rate heroes
bottom_heroes_html = ""
bottom_heroes = hero_importance[hero_importance['pick_count'] >= 10].nsmallest(10, 'win_rate')
for _, row in bottom_heroes.iterrows():
    bottom_heroes_html += create_hero_card(
        row['hero_id'], row['hero_name'], row['win_rate'], 
        int(row['pick_count']), row['importance'],
        hero_data.get(row['hero_id'], {})
    )

# Most influential heroes
influential_html = ""
influential = hero_importance.nlargest(10, 'importance')
for _, row in influential.iterrows():
    influential_html += create_hero_card(
        row['hero_id'], row['hero_name'], row['win_rate'] if pd.notna(row['win_rate']) else 0.5, 
        int(row['pick_count']) if pd.notna(row['pick_count']) else 0, row['importance'],
        hero_data.get(row['hero_id'], {})
    )

# Full HTML template
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dota 2 Hero Win Rate Analysis</title>
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
            max-width: 1400px;
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
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
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
        
        .section-title .badge {{
            font-size: 0.8rem;
            padding: 4px 12px;
            border-radius: 20px;
            margin-left: auto;
        }}
        
        .badge-good {{
            background: rgba(76, 175, 80, 0.2);
            color: #4CAF50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }}
        
        .badge-bad {{
            background: rgba(244, 67, 54, 0.2);
            color: #F44336;
            border: 1px solid rgba(244, 67, 54, 0.3);
        }}
        
        .badge-neutral {{
            background: rgba(255, 215, 0, 0.2);
            color: #ffd700;
            border: 1px solid rgba(255, 215, 0, 0.3);
        }}
        
        /* Feature Bars */
        .features-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .feature-item {{
            background: rgba(0, 0, 0, 0.2);
            padding: 12px 18px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .feature-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 0.9rem;
        }}
        
        .feature-name {{
            color: #ccc;
        }}
        
        .feature-value {{
            color: #ffd700;
            font-weight: 600;
        }}
        
        .feature-bar-bg {{
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .feature-bar {{
            height: 100%;
            background: linear-gradient(90deg, #ff8c00, #ffd700);
            border-radius: 3px;
        }}
        
        /* Hero Grid */
        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 16px;
        }}
        
        .hero-card {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .hero-card:hover {{
            transform: scale(1.05);
            border-color: rgba(255, 215, 0, 0.5);
            box-shadow: 0 5px 20px rgba(255, 215, 0, 0.2);
        }}
        
        .hero-image-wrapper {{
            width: 80px;
            height: 45px;
            margin: 0 auto 8px;
        }}
        
        .hero-image-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
        }}
        
        .hero-info {{
            min-height: 50px;
        }}
        
        .hero-name {{
            display: block;
            font-size: 0.8rem;
            color: #ddd;
            font-weight: 500;
            margin-bottom: 4px;
        }}
        
        .hero-stats {{
            display: flex;
            justify-content: center;
            gap: 8px;
            font-size: 0.75rem;
        }}
        
        .hero-stats .win-rate {{
            font-weight: 600;
        }}
        
        .hero-stats .picks {{
            color: #888;
        }}
        
        /* Insight Box */
        .insight-box {{
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid rgba(255, 215, 0, 0.2);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .insight-box h4 {{
            color: #ffd700;
            margin-bottom: 10px;
        }}
        
        .insight-box p {{
            color: #ccc;
            line-height: 1.6;
            font-size: 0.95rem;
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
                grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Dota 2 Hero Win Rate Analysis</h1>
            <p class="subtitle">Machine Learning Model: Random Forest Classifier</p>
        </div>
        
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_matches:,}</div>
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
                <div class="stat-label">Model Trees</div>
            </div>
        </div>
        
        <!-- Feature Importance -->
        <div class="section">
            <h2 class="section-title">
                Feature Importance
                <span class="badge badge-neutral">What affects win rate?</span>
            </h2>
            <div class="features-list">
                {feature_bars}
            </div>
            <div class="insight-box">
                <h4>Key Insight</h4>
                <p>Duration and rank tier are the strongest predictors. Hero selection features have moderate importance, suggesting team composition matters but individual hero performance varies with skill level.</p>
            </div>
        </div>
        
        <!-- Top Win Rate Heroes -->
        <div class="section">
            <h2 class="section-title">
                Highest Win Rate Heroes
                <span class="badge badge-good">Radiant Pick Advantage</span>
            </h2>
            <div class="hero-grid">
                {top_heroes_html}
            </div>
        </div>
        
        <!-- Bottom Win Rate Heroes -->
        <div class="section">
            <h2 class="section-title">
                Lowest Win Rate Heroes
                <span class="badge badge-bad">Requires More Skill</span>
            </h2>
            <div class="hero-grid">
                {bottom_heroes_html}
            </div>
            <div class="insight-box">
                <h4>Note on Low Win Rate Heroes</h4>
                <p>Heroes like Chen, Io, and Batrider require team coordination and specific strategies. Their low win rate doesn't mean they're weak - they have high skill ceilings and work best in coordinated stacks.</p>
            </div>
        </div>
        
        <!-- Most Influential Heroes -->
        <div class="section">
            <h2 class="section-title">
                Most Influential Heroes
                <span class="badge badge-neutral">Model Focus Areas</span>
            </h2>
            <div class="hero-grid">
                {influential_html}
            </div>
            <div class="insight-box">
                <h4>Model Perspective</h4>
                <p>These heroes have the most impact on the model's predictions. When these heroes appear in a match, they significantly shift the predicted outcome probability.</p>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Data Source: OpenDota API | Model: Random Forest Classifier</p>
        </div>
    </div>
</body>
</html>"""

# Write final HTML
with open('visualization/report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Report generated: visualization/report.html")
print(f"- {total_matches:,} matches analyzed")
print(f"- {len(hero_data)} heroes")
