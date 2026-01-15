import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import spacy
from tqdm import tqdm
from collections import Counter
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
nlp = spacy.load("en_core_web_sm")

def plot_gender_distribution_top_subreddits(df, target, gender_col="gender", top_n=10):
    # Merge df with target on author
    merged = df.merge(target[['author', gender_col]], on='author', how='left')
    top_subreddits = (
        merged['subreddit']
        .value_counts()
        .head(top_n)
        .index
    )
    filtered = merged[merged['subreddit'].isin(top_subreddits)]
    filtered["gender_label"] = filtered[gender_col].map({0: "male", 1: "female"})
    plt.figure(figsize=(12, 7))
    sns.countplot(
        data=filtered,
        y="subreddit",
        hue="gender_label",
        palette={"male": "#1f77b4", "female": "#ff69b4"}
    )

    plt.title(f"Gender distribution in the top {top_n} subreddits")
    plt.xlabel("Count")
    plt.ylabel("Subreddit")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

def plot_user_subreddit_heatmap(df, top_users=20, top_subreddits=30):
    top_user_list = (
        df['author']
        .value_counts()
        .head(top_users)
        .index
    )
    top_sub_list = (
        df['subreddit']
        .value_counts()
        .head(top_subreddits)
        .index
    )
    # filter dataset
    filtered = df[
        df['author'].isin(top_user_list) &
        df['subreddit'].isin(top_sub_list)
    ]
    # table: rows = users, columns = subreddits
    pivot = (
        filtered
        .pivot_table(
            index='author',
            columns='subreddit',
            values='body',
            aggfunc='count',
            fill_value=0
        )
    )

    plt.figure(figsize=(18, 10))
    sns.heatmap(
        pivot,
        cmap="viridis",
        linewidths=0.4,
        linecolor="gray"
    )
    plt.title("Activity Heatmap: Top Users × Top Subreddits")
    plt.xlabel("Subreddit")
    plt.ylabel("User")
    plt.tight_layout()
    plt.show()

def process_text_full(text_series, batch_size=2000):
    clean_texts = []

    total_docs = len(text_series)

    # tqdm show the process bar
    for doc in tqdm(nlp.pipe(text_series, batch_size=batch_size), total=total_docs, desc="Processing"):

        tokens = []
        for token in doc:
            # 1. Filtering Stop Words e punctation (b)
            if not token.is_stop and not token.is_punct and not token.is_space:
                # 2. Take the lemma using spaCy (c)
                tokens.append(token.lemma_)

        clean_texts.append(" ".join(tokens))

    return clean_texts

def get_top_words(body, n=20):

    all_body = ' '.join(body.fillna(''))
    words = all_body.split()
    return pd.DataFrame(Counter(words).most_common(n), columns=['word', 'count'])

def detect_language(text):
    try:
        text = str(text).strip()
        if not text:
            return "unknown"
        return detect(text)
    except:
        return "unknown"
    
def add_language_column(df, text_column='body_clean'):
    """
    Adds a 'language' column to the DataFrame by detecting the language
    of each entry in the specified text column.
    """
    df = df.copy()
    df['language'] = df[text_column].apply(detect_language)
    return df

def get_top_languages(df, column='language', top_n=4):
    counts = df[column].value_counts()
    top = counts.head(top_n)
    other = counts.iloc[top_n:].sum()

    result = top.to_frame().reset_index()
    result.columns = ['language', 'count']

    if other > 0:
        result.loc[len(result)] = ['other', other]

    return result