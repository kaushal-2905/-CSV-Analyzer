import pandas as pd

def read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        print("⚠️ Warning: UTF-8 decoding failed. Falling back to ISO-8859-1.")
        return pd.read_csv(file_path, encoding='ISO-8859-1')


def check_nulls(df):
    return df.isnull().sum()

def check_duplicates(df):
    return df.duplicated().sum()

def get_dtypes(df):
    return df.dtypes

def get_head(df, n=10):
    return df.head(n)

def remove_duplicates(df):
    return df.drop_duplicates()

def remove_nulls(df):
    return df.dropna()
