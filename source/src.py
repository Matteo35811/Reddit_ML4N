import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import spacy
from tqdm import tqdm
from collections import Counter
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
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

def stratified_split(X, Y):
    #split 80-10-10
    X_temp, X_test, y_temp, y_test = train_test_split(X, Y, stratify=Y, train_size=0.90,random_state=16)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, train_size=8/9, random_state=16)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_tfidf_sec1(corpus, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english"
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out(), vectorizer

def build_tfidf(x_train, x_val, x_test):
    vectorizer = TfidfVectorizer(
        ngram_range=(1,1),
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(x_train)
    X_val = vectorizer.transform(x_val)
    X_test = vectorizer.transform(x_test)

def build_bow(x_train, x_val, x_test):
    bow = CountVectorizer(
        stop_words="english",
        ngram_range=(1,1)
    )
    X_train =bow.fit_transform(x_train)
    X_val =bow.transform(x_val)
    X_test =bow.transform(x_test)
    return X_train, X_val, X_test


# def quantile_reduction

# def truncated_svd