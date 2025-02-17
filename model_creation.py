import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


logreg_param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear'],
        'max_iter': [100, 200, 500]
    },
    {
        'penalty': ['l2', 'none'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['lbfgs'],
        'max_iter': [100, 200, 500]
    },
    {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['saga'],
        'max_iter': [100, 200, 500]
    }
]

svc_param_grid = {
    'C': [0.01, 0.1, 1.0, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'degree': [2, 3, 4]
}

clf_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['Sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']

}

def log_reg(data, target_col):

    df = pd.DataFrame(data)
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=str_cols)
    x= df.drop(columns=target_col)
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)

    grid_search_logreg = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=logreg_param_grid,
        scoring='f1',
        cv=5
    )

    grid_search_logreg.fit(x_train, y_train)

    y_pred = grid_search_logreg.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    df_final = df.copy()
    df_final['prediction'] = grid_search_logreg.predict(df.drop(columns=target_col))
    df_final['accuracy'] = (df_final[target_col] == df_final['prediction']).astype(int)

    return df_final.to_dict('records'), f1, recall, precision


def svm_svc(data, target_col):
    df = pd.DataFrame(data)
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=str_cols)
    x = df.drop(columns=target_col)
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)

    grid_search_svc = GridSearchCV(
        estimator=SVC(),
        param_grid=svc_param_grid,
        scoring='f1',
        cv=5
    )

    grid_search_svc.fit(x_train, y_train)

    y_pred = grid_search_svc.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    df_final = df.copy()
    df_final['prediction'] = grid_search_svc.predict(df.drop(columns=target_col))
    df_final['accuracy'] = (df_final[target_col] == df_final['prediction']).astype(int)

    return df_final.to_dict('records'), f1, recall, precision


def clf(data, target_col):

    df = pd.DataFrame(data)
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=str_cols)
    x = df.drop(columns=target_col)
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)

    grid_search_clf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=clf_param_grid,
        scoring='f1',
        cv=5
    )

    grid_search_clf.fit(x_train, y_train)

    y_pred = grid_search_clf.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    df_final = df.copy()
    df_final['prediction'] = grid_search_clf.predict(df.drop(columns=target_col))
    df_final['accuracy'] = (df_final[target_col] == df_final['prediction']).astype(int)

    return df_final.to_dict('records'), f1, recall, precision