from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target

def linear_regression(df, features, target):
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_, model.intercept_

# def decision_tree_classification(df, features, target):
#     X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)
#     clf = DecisionTreeClassifier()
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     return accuracy_score(y_test, predictions)

def decision_tree_classification(df, features, target):
    X = df[features]
    y = df[target]

    # Check if target is categorical (classification only supports discrete labels)
    target_type = type_of_target(y)
    if target_type not in ['binary', 'multiclass']:
        raise ValueError("Target must be categorical for classification.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions)

def kmeans_clustering(df, features, k=3):
    model = KMeans(n_clusters=k)
    model.fit(df[features])
    return model.labels_

def detect_outliers(df, features):
    iso = IsolationForest(contamination=0.1)
    preds = iso.fit_predict(df[features])
    return preds

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def linear_regression_metrics(df, col1, col2):
    X = df[[col1]]
    y = df[col2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }

def multiple_linear_regression_metrics(df, target):
    X = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }

def polynomial_regression_metrics(df, target, degree=2):
    X = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    return {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }
