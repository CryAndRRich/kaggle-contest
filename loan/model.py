# Updated on April 23, 2025, 05:22 PM
# Public Score: 0.53957
# Rank: 45/217

import os
from typing import List, Tuple

import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

from preprocess import Data

class LoanModel():
    """
    Loan default prediction using CatBoostClassifier with Bayesian hyperparameter tuning,
    early stopping on AUC, and F1-threshold tuning
    """
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        """
        Initialize Loan model, load data, and split into train/test sets

        Parameters:
            data_path: Relative path to the data directory from the working directory
            data_processed_path: CSV filename of the preprocessed data
        """
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        # Split into train and test by Id
        self.train_data = data[data["Id"] <= 7499].copy()
        self.test_data = data[data["Id"] > 7499].copy()

        self.test_ids = self.test_data["Id"]
        self.X_train = self.train_data.drop(columns=["Credit Default", "Id"])
        self.y_train = self.train_data["Credit Default"]
        self.X_test = self.test_data.drop(columns=["Credit Default", "Id"])

        # Identify numeric and categorical features
        self.numerical_features = (
            self.X_train.select_dtypes(include=["int64", "float64"])
            .columns.tolist()
        )
        self.categorical_features = (
            self.X_train.select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        # Build preprocessing pipeline and optimize CatBoost params
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Construct preprocessing pipelines for numeric and categorical features,
        perform Bayesian optimization of CatBoost hyperparameters, and
        assemble the final Pipeline
        """
        # Numeric imputation and scaling
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Categorical imputation and one-hot encoding
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", num_transformer, self.numerical_features),
            ("cat", cat_transformer, self.categorical_features)
        ])

        # Perform Bayesian optimization on F1 via cross-validation
        self._bayes_opt()

        # Final pipeline combining preprocessing and CatBoost with optimized params
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("cat", CatBoostClassifier(
                **self.best_params_cat,
                random_seed=42,
                eval_metric="AUC",
                verbose=False
            ))
        ])

    def _bayes_opt(self) -> None:
        """
        Execute Bayesian hyperparameter optimization on CatBoostClassifier,
        maximizing cross-validated F1 score
        """
        def cat_cv(iterations: float,
                   learning_rate: float,
                   depth: float,
                   l2_leaf_reg: float,
                   subsample: float) -> float:
            """
            Objective function for Bayesian optimization: returns mean F1 score
            from 3-fold cross-validation

            Parameters:
                iterations: Number of boosting iterations
                learning_rate: Learning rate for boosting
                depth: Depth of each decision tree
                l2_leaf_reg: L2 regularization coefficient
                subsample: Fraction of samples to subsample

            Returns:
                float: Mean F1 score over cross-validation splits
            """
            iters = int(iterations)
            d = int(depth)
            model = CatBoostClassifier(
                iterations=iters,
                learning_rate=learning_rate,
                depth=d,
                l2_leaf_reg=l2_leaf_reg,
                subsample=subsample,
                random_seed=42,
                eval_metric="F1",
                verbose=False
            )
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("cat", model)
            ])
            return cross_val_score(
                pipe, self.X_train, self.y_train,
                cv=3, scoring="f1"
            ).mean()

        pbounds = {
            "iterations": (100, 500),
            "learning_rate": (0.01, 0.3),
            "depth": (4, 10),
            "l2_leaf_reg": (1, 10),
            "subsample": (0.6, 1.0)
        }

        optimizer = BayesianOptimization(
            f=cat_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=15)

        params = optimizer.max["params"]
        # Cast integer params
        params["iterations"] = int(params["iterations"])
        params["depth"] = int(params["depth"])
        self.best_params_cat = params
        print("Best CatBoost params:", self.best_params_cat)

    def train(self) -> None:
        """
        Train the CatBoost model with early stopping on AUC and tune
        the decision threshold to maximize F1 score on a validation set
        """
        # Hold-out split for early stopping and threshold tuning
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.1,
            stratify=self.y_train,
            random_state=42
        )
        # Fit preprocessor on training data
        self.preprocessor.fit(X_tr, y_tr)
        X_tr_enc = self.preprocessor.transform(X_tr)
        X_val_enc = self.preprocessor.transform(X_val)

        # Train CatBoost natively to utilize early stopping
        model = CatBoostClassifier(
            **self.best_params_cat,
            random_seed=42,
            eval_metric="AUC"
        )
        model.fit(
            X_tr_enc, y_tr,
            eval_set=(X_val_enc, y_val),
            early_stopping_rounds=20,
            verbose=50
        )

        # Optimize threshold on validation set to maximize F1
        val_probs = model.predict_proba(X_val_enc)[:, 1]
        best_thresh, best_f1 = self._threshold_optim(y_val, val_probs)
        print(f"Best F1 = {best_f1:.4f} at threshold = {best_thresh}")

        self.model = model
        self.best_thresh = best_thresh

    def _threshold_optim(self,
                         y_true: pd.Series,
                         y_prob: pd.Series|List|Tuple) -> Tuple[float, float]:
        """
        Find the probability threshold that maximizes F1 score

        Parameters:
            y_true: True binary labels
            y_prob: Predicted probabilities for the positive class

        Returns:
            Tuple[float, float]: Best threshold and corresponding F1 score
        """
        def objective(thresh: float) -> float:
            preds = (y_prob > thresh).astype(int)
            return -f1_score(y_true, preds)

        result = minimize_scalar(
            objective,
            bounds=(0.1, 0.9),
            method="bounded"
        )
        return result.x, -result.fun

    def predict(self) -> None:
        """
        Generate predictions on the test set using the trained model and tuned
        threshold, then save to CSV
        """
        # Preprocess test data
        X_test_enc = self.preprocessor.transform(self.X_test)
        # Predict probabilities and apply threshold
        probs = self.model.predict_proba(X_test_enc)[:, 1]
        preds = (probs > self.best_thresh).astype(int)

        # Save submission
        self.submission = pd.DataFrame({
            "Id": self.test_ids,
            "Credit Default": preds
        })
        outfile = os.path.join(self.data_dir, "loan_submission.csv")
        self.submission.to_csv(outfile, index=False)
        print("Submission saved to loan_submission.csv!")


if __name__ == "__main__":
    data_path = "loan/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = LoanModel(data_path, data_processed_path)
    model.train()
    model.predict()

# Data saved to processed_data.csv!
# |   iter    |  target   |   depth   | iterat... | l2_lea... | learni... | subsample |
# -------------------------------------------------------------------------------------
# | 1         | 0.4681    | 6.247     | 480.3     | 7.588     | 0.1836    | 0.6624    |
# | 2         | 0.433     | 4.936     | 123.2     | 8.796     | 0.1843    | 0.8832    |
# | 3         | 0.4442    | 4.124     | 488.0     | 8.492     | 0.07158   | 0.6727    |
# | 4         | 0.4534    | 5.1       | 221.7     | 5.723     | 0.1353    | 0.7165    |
# | 5         | 0.4444    | 7.671     | 155.8     | 3.629     | 0.1162    | 0.7824    |
# | 6         | 0.4586    | 6.125     | 480.8     | 7.292     | 0.2289    | 0.903     |
# | 7         | 0.4549    | 6.73      | 479.5     | 7.841     | 0.09468   | 0.8499    |
# | 8         | 0.4541    | 6.168     | 480.5     | 8.49      | 0.0863    | 0.7573    |
# | 9         | 0.4558    | 6.207     | 480.2     | 6.578     | 0.0877    | 0.686     |
# | 10        | 0.4497    | 5.375     | 479.3     | 7.754     | 0.05316   | 0.8916    |
# | 11        | 0.4521    | 7.24      | 480.6     | 7.148     | 0.08282   | 0.9932    |
# | 12        | 0.4498    | 4.453     | 424.8     | 5.2       | 0.09868   | 0.7195    |
# | 13        | 0.4656    | 4.495     | 461.5     | 6.344     | 0.2274    | 0.8992    |
# | 14        | 0.4598    | 8.369     | 367.0     | 9.951     | 0.1279    | 0.8155    |
# | 15        | 0.4357    | 4.802     | 394.9     | 3.712     | 0.06496   | 0.6907    |
# | 16        | 0.4656    | 5.453     | 461.9     | 6.033     | 0.2072    | 0.7165    |
# | 17        | 0.4609    | 5.176     | 461.9     | 6.824     | 0.1912    | 0.8874    |
# | 18        | 0.4396    | 4.742     | 462.6     | 6.034     | 0.08437   | 0.6191    |
# | 19        | 0.4544    | 7.034     | 304.7     | 8.123     | 0.1058    | 0.969     |
# | 20        | 0.4719    | 6.277     | 462.2     | 6.366     | 0.2715    | 0.702     |
# =====================================================================================
# Best CatBoost params: {'depth': 6, 'iterations': 462, 'l2_leaf_reg': 6.3662766490055, 'learning_rate': 0.2715303162227133, 'subsample': 0.7019637567515548}
# 0:      test: 0.7146198 best: 0.7146198 (0)     total: 23ms     remaining: 10.6s
# Stopped by overfitting detector  (20 iterations wait)

# bestTest = 0.7488943014
# bestIteration = 24

# Shrink model to first 25 iterations.
# Best F1 = 0.5345 at threshold = 0.22703349868496658
# Submission saved to loan_submission.csv!
