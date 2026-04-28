from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

def create_and_save_dataset():
    print("Завантаження повного датасету 20 Newsgroups...")
    
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'target': newsgroups.target,
        'target_name': [newsgroups.target_names[i]
                        for i in newsgroups.target]
    })
    
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/dataset.csv', index=False)
    
    print(f"Збережено: data/raw/dataset.csv")
    print(f"Розмір: {df.shape}")
    print(f"Кількість класів: {df['target'].nunique()}")
    print(f"Класи: {df['target_name'].unique()}")
    return df

if __name__ == "__main__":
    create_and_save_dataset()