from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import logging

RANDOM_SEED = 42


def train_logistic_regression(X_train, y_train):
    logging.info("Training Logistic Regression model...")
    lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000)
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    }
    grid = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=0)
    model = grid.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    logging.info("Training Decision Tree model...")
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    param_grid = {
        "max_depth": [10, 20, None],
        "criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=0)
    model = grid.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    logging.info("Training SVM model...")
    svm = SVC(class_weight="balanced", random_state=RANDOM_SEED)
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
    }
    grid = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=0)
    model = grid.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    logging.info("Training Random Forest model...")
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [20, 30, None],
        "criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=0)
    model = grid.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    logging.info("Training Gradient Boosting model...")
    gb = GradientBoostingClassifier(random_state=RANDOM_SEED)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(gb, param_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=0)
    model = grid.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    logging.info("Training XGBoost model...")
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [6, 7],
        "learning_rate": [0.05, 0.1],
        "gamma": [0, 0.1],
        "min_child_weight": [1, 3],
    }
    grid = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=0)
    model = grid.fit(X_train, y_train)
    return model
