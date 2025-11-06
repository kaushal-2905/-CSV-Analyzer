import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for server environments

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def save_plot(fig, name):
    path = os.path.join("analyzer/static", f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return f"{name}.png"

def heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return None
    fig = plt.figure()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='tab20')
    return save_plot(fig, "heatmap")

def histogram(df, column):
    fig = plt.figure()
    df[column].hist()
    return save_plot(fig, f"hist_{column}")

def bar_chart(df, column):
    fig = plt.figure()

    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        # Categorical column — plot value counts directly
        df[column].value_counts().plot(kind='bar', color='skyblue')
        plt.ylabel("Count")
        plt.title(f"Bar Chart of {column}")
    else:
        # Numerical column — bin the data before plotting
        binned = pd.cut(df[column], bins=10)  # You can increase/decrease bins
        binned.value_counts().sort_index().plot(kind='bar', color='orange')
        plt.xlabel(f"{column} (binned)")
        plt.ylabel("Frequency")
        plt.title(f"Binned Bar Chart of {column}")

    plt.xticks(rotation=45)
    plt.tight_layout()
    return save_plot(fig, f"bar_{column}")

def scatter_plot(df, col1, col2):
    fig = plt.figure()
    sns.scatterplot(x=df[col1], y=df[col2])
    return save_plot(fig, f"scatter_{col1}_{col2}")

def box_plot(df, column):
    fig = plt.figure()
    sns.boxplot(y=df[column])
    return save_plot(fig, f"box_{column}")

def parallel_coordinates_plot(df, target_col):
    # Select numeric columns only
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    feature_cols = [col for col in numeric_columns if col != target_col]

    # Drop rows with NaNs in relevant columns
    selected_df = df[feature_cols + [target_col]].dropna()

    # Create larger high-resolution figure
    fig = plt.figure(figsize=(16, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    # Plot the parallel coordinates
    parallel_coordinates(selected_df, class_column=target_col, colormap=plt.cm.Set2, ax=ax)

    # Improve layout and readability
    plt.title(f"Parallel Coordinates Plot (Target: {target_col})"  , fontsize=30)
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout(pad=3.0)

    return save_plot(fig, f"parallel_{target_col}")

# def line_plot(df, x_col):
#     import matplotlib.pyplot as plt

#     # Drop rows with NaN in x_col
#     df = df.dropna(subset=[x_col])

#     # Make sure the x_col exists
#     if x_col not in df.columns:
#         return None

#     # Select all numeric columns except x_col
#     numeric_df = df.select_dtypes(include='number')
#     if x_col in numeric_df.columns:
#         numeric_df = numeric_df.drop(columns=[x_col])

#     # ✅ Ensure we have at least one numeric column to plot
#     if numeric_df.empty:
#         return None

#     # Create the figure
#     fig, ax = plt.subplots()

#     for col in numeric_df.columns:
#         ax.plot(df[x_col], df[col], label=col)

#     ax.set_xlabel(x_col)
#     ax.set_ylabel("Values")
#     ax.set_title(f"Line Plot vs {x_col}")
#     ax.legend()
#     ax.grid(True)

#     return save_plot(fig, f"line_plot_{x_col}")

