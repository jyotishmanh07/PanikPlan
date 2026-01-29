import pandas as pd
from pathlib import Path

def load_and_clean_data(relative_path):
    base_path = Path(__file__).parent.parent.parent
    full_path = (base_path / relative_path).resolve()
    
    df = pd.read_csv(full_path)
    
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    price_map = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
    df['price_score'] = df['price_range'].map(price_map).fillna(1)
    
    df['cuisine_style'] = df['cuisine_style'].fillna("['Unknown']")
    df['main_cuisine'] = df['cuisine_style'].apply(
        lambda x: eval(x)[0] if x.startswith('[') else x
    )
    
    # Define the specific cuisines we want to track
    top_cuisines = ['European', 'Italian', 'French', 'Mediterranean', 'Vegetarian Friendly']
    for cuisine in top_cuisines:
        col_name = f"is_{cuisine.lower().replace(' ', '_')}"
        df[col_name] = (df['main_cuisine'] == cuisine).astype(int)
    
    df['is_unknown'] = (df['main_cuisine'] == 'Unknown').astype(int)

    features = [
        'price_score', 
        'number_of_reviews', 
        'is_european', 
        'is_italian', 
        'is_french', 
        'is_mediterranean', 
        'is_vegetarian_friendly', 
        'is_unknown'
    ]
    target = 'rating'
    
    df_clean = df[features + [target]].dropna()
    return df_clean[features], df_clean[target]