# Updated on March 15, 2025, 10:51 AM
# Public Score: 0.79665
# Rank: 837/15256

import os
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from bayes_opt import BayesianOptimization

from preprocess import Data


def to_numeric_func(x: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the input DataFrame to numeric, coercing errors.

    Parameters:
        x: The input DataFrame.

    Returns:
        A DataFrame with numeric conversion applied.
    """
    return x.apply(pd.to_numeric, errors='coerce')


class TitanicModel:
    """
    A class to build an ensemble model to predict Titanic survival using XGBoost and CatBoost+PCA.
    """
    def __init__(self, 
                 datasets_path: str, 
                 data_processed_path: str,
                 checkpoints_path: str) -> None:
        """
        Initialize the model with the paths for datasets, processed data, and checkpoints.

        Parameters:
            datasets_path: Path to the directory containing the datasets.
            data_processed_path: Path to the processed data CSV file.
            checkpoints_path: Path to the directory containing the checkpoints.
        """
        self.checkpoints_path = checkpoints_path
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)
        data_path = os.path.join(self.datasets_dir, data_processed_path)
        data = pd.read_csv(data_path)

        # Split data into train and test sets based on PassengerId
        self.train_data = data[data["PassengerId"] <= 891]
        self.test_data = data[data["PassengerId"] > 891]

        # Define features (excluding PassengerId and Survived)
        self.features = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "IsAdult", 
            "FamSize", "hasNanny", "TicketNumber", "TicketLetter", "CabinLet", "CabinNum"
        ]

        # Prepare training data
        self.X = self.train_data[self.features]
        self.y = self.train_data["Survived"]

        # Prepare test data
        self.X_test = self.test_data[self.features]
        self.test_id = self.test_data["PassengerId"]

        # Define numeric and categorical features for the common (XGBoost) pipeline
        self.numeric_features = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "IsAdult",
            "FamSize", "hasNanny", "TicketNumber", "CabinNum"
        ]
        self.categorical_features = [col for col in self.features if col not in self.numeric_features]
        print("Numeric features:", self.numeric_features)
        print("Categorical features:", self.categorical_features)

        self._build_pineline()

    def _build_pineline(self) -> None:
        """
        Build the preprocessing pipeline and the model pipelines for XGBoost and CatBoost+PCA.
        """
        # Build numeric transformer pipeline for XGBoost using the named function instead of a lambda.
        self.numeric_transformer = Pipeline(steps=[
            ('to_numeric', FunctionTransformer(to_numeric_func)),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Build categorical transformer pipeline for XGBoost
        self.categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", self.numeric_transformer, self.numeric_features),
            ("cat", self.categorical_transformer, self.categorical_features)
        ])

        # Build XGBoost pipeline (without PCA)
        self.pipeline_xgb = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("xgb", XGBClassifier(n_estimators=20, random_state=71, eval_metric="logloss"))
        ])

        # Build CatBoost+PCA pipeline
        # Define PCA columns (apply PCA to these numeric columns)
        pca_columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        # The remaining columns will be passed through unchanged.
        self.catboost_remainder = [col for col in self.features if col not in pca_columns]

        # Build a preprocessor for CatBoost that applies PCA on pca_columns and passes through the rest.
        self.preprocessor_cat_pca = ColumnTransformer(
            transformers=[
                ('pca', PCA(n_components=3), pca_columns)
            ],
            remainder='passthrough'
        )

        self._bayes_opt()

        # Build final CatBoost+PCA pipeline with tuned hyperparameters
        self.pipeline_cat = Pipeline(steps=[
            ('preprocessing', self.preprocessor_cat_pca),
            ('cat', CatBoostClassifier(
                **self.best_params_cat,
                random_seed=71,
                verbose=False,
                cat_features=self.cat_feature_indices
            ))
        ])

    def _bayes_opt(self) -> None:
        """
        Use Bayesian Optimization to tune the hyperparameters for the CatBoost+PCA model.
        """
        # Fit the CatBoost+PCA preprocessor to determine categorical feature indices.
        X_cat_pca_transformed = self.preprocessor_cat_pca.fit_transform(self.X)
        # Assume that the last len(catboost_remainder) columns are the non-PCA (categorical) features
        self.cat_feature_indices = list(range(
            X_cat_pca_transformed.shape[1] - len(self.catboost_remainder),
            X_cat_pca_transformed.shape[1]
        ))

        def cat_cv(iterations: float, learning_rate: float, depth: float) -> float:
            """
            Evaluate the CatBoost+PCA pipeline using cross-validation.

            Parameters:
                iterations: Number of boosting rounds (will be cast to int).
                learning_rate: Learning rate.
                depth: Tree depth (will be cast to int).

            Returns:
                float: Mean ROC AUC score from cross-validation.
            """
            iterations_int = int(iterations)
            depth_int = int(depth)
            model = CatBoostClassifier(
                iterations=iterations_int,
                learning_rate=learning_rate,
                depth=depth_int,
                random_seed=71,
                verbose=False,
                cat_features=self.cat_feature_indices
            )
            pipe = Pipeline(steps=[
                ('preprocessing', self.preprocessor_cat_pca),
                ('cat', model)
            ])
            scores = cross_val_score(pipe, self.X, self.y, cv=3, scoring="roc_auc")
            return scores.mean()

        # Set up Bayesian Optimization with bounds for the hyperparameters
        cat_bo = BayesianOptimization(
            f=cat_cv,
            pbounds={
                'iterations': (100, 500),
                'learning_rate': (0.01, 0.3),
                'depth': (3, 8)
            },
            random_state=71,
            verbose=2
        )
        cat_bo.maximize(init_points=5, n_iter=15)

        self.best_params_cat = cat_bo.max['params']
        self.best_params_cat['iterations'] = int(self.best_params_cat['iterations'])
        self.best_params_cat['depth'] = int(self.best_params_cat['depth'])
        print("Best CatBoost parameters:", self.best_params_cat)

    def train(self) -> None:
        """
        Train the XGBoost and CatBoost+PCA pipelines, then find the best ensemble weights based on validation data.
        """
        # Split the data into training and validation sets for ensemble tuning
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Train both pipelines on the training split
        self.pipeline_xgb.fit(X_train_split, y_train_split)
        self.pipeline_cat.fit(X_train_split, y_train_split)

        # Predict probabilities on the validation set for each model
        val_pred_xgb = self.pipeline_xgb.predict_proba(X_val)[:, 1]
        val_pred_cat = self.pipeline_cat.predict_proba(X_val)[:, 1]

        # Find the best ensemble weights (weighted average) based on validation accuracy
        self.best_w_xgb, self.best_w_cat = self.find_best_ensemble_weights(
            y_val, val_pred_xgb, val_pred_cat, step=0.01
        )
        print("Best ensemble weights: XGB: {:.2f}, CatBoost+PCA: {:.2f}".format(self.best_w_xgb, self.best_w_cat))

    def find_best_ensemble_weights(self, 
                                   y_true: np.ndarray,
                                   y_pred_xgb: np.ndarray,
                                   y_pred_cat: np.ndarray,
                                   step: float = 0.01) -> Tuple[float, float]:
        """
        Find the best ensemble weights for combining predictions from two models (weighted average)
        based on validation accuracy.

        Parameters:
            y_true: True labels of the validation set.
            y_pred_xgb: Predicted probabilities from the XGBoost model.
            y_pred_cat: Predicted probabilities from the CatBoost+PCA model.
            step: Step size for iterating through weights from 0 to 1. Defaults to 0.01.

        Returns:
            Tuple[float, float]: Weights for XGBoost and CatBoost (corresponding to w and 1-w).
        """
        best_w = 0.0
        best_acc = 0.0
        for w in np.arange(0, 1 + step, step):
            ensemble_prob = w * y_pred_xgb + (1 - w) * y_pred_cat
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
            acc = accuracy_score(y_true, ensemble_pred)
            if acc > best_acc:
                best_w, best_acc = w, acc
        return best_w, 1 - best_w

    def save_checkpoint(self) -> None:
        """
        Save the trained pipelines for XGBoost and CatBoost+PCA as checkpoints to files.
        """
        checkpoint_xgb_path = os.path.join(self.checkpoints_path, "checkpoint_xgb.pth")
        checkpoint_cat_path = os.path.join(self.checkpoints_path, "checkpoint_cat.pth")
        joblib.dump(self.pipeline_xgb, checkpoint_xgb_path)
        joblib.dump(self.pipeline_cat, checkpoint_cat_path)
        print("Checkpoints saved to", checkpoint_xgb_path, "and", checkpoint_cat_path)

    def predict(self) -> None:
        """
        Predict on the test set using an ensemble of the two models.
        If checkpoint files exist, load the saved models before predicting.
        Then, save the submission file.
        """
        checkpoint_xgb_path = os.path.join(self.checkpoints_path, "checkpoint_xgb.pth")
        checkpoint_cat_path = os.path.join(self.checkpoints_path, "checkpoint_cat.tph")
        try:
            if os.path.exists(checkpoint_xgb_path) and os.path.exists(checkpoint_cat_path):
                self.pipeline_xgb = joblib.load(checkpoint_xgb_path)
                self.pipeline_cat = joblib.load(checkpoint_cat_path)
                print("Loaded checkpoints from files.")
            else:
                print("Checkpoints not found. Using current pipelines.")
        except Exception as e:
            print("Error loading checkpoints:", e)

        # Predict probabilities on the test set for each model
        test_pred_xgb = self.pipeline_xgb.predict_proba(self.X_test)[:, 1]
        test_pred_cat = self.pipeline_cat.predict_proba(self.X_test)[:, 1]

        ensemble_test_prob = self.best_w_xgb * test_pred_xgb + self.best_w_cat * test_pred_cat
        final_predictions = (ensemble_test_prob > 0.5).astype(int)

        self.submission = pd.DataFrame({
            "PassengerId": self.test_id,
            "Survived": final_predictions
        })

        output_file = os.path.join(self.datasets_dir, "titanic_submission.csv")
        self.submission.to_csv(output_file, index=False)
        print("Submission saved to titanic_submission.csv!")

if __name__ == "__main__":
    datasets_path = "titanic/datasets"
    checkpoints_path = "titanic/checkpoints"

    data = Data(datasets_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = TitanicModel(datasets_path, data_processed_path, checkpoints_path)
    model.train()
    # Optionally, save checkpoints after training
    model.save_checkpoint()
    model.predict()


# Data saved to processed_data.csv!

# Numeric features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'IsAdult', 'FamSize', 'hasNanny', 'TicketNumber', 'CabinNum']
# Categorical features: ['Embarked', 'TicketLetter', 'CabinLet']
# |   iter    |  target   |   depth   | iterat... | learni... | 
# ------------------------------------------------------------- 
# | 1         | 0.7696    | 3.928     | 254.6     | 0.2512    |
# | 2         | 0.758     | 3.993     | 466.6     | 0.2386    |
# | 3         | 0.7648    | 6.613     | 498.3     | 0.2552    |
# | 4         | 0.7605    | 4.169     | 229.0     | 0.2178    |
# | 5         | 0.7941    | 3.514     | 340.0     | 0.04952   |
# | 6         | 0.7766    | 4.278     | 339.7     | 0.1243    |
# | 7         | 0.7705    | 3.04      | 339.8     | 0.163     |
# | 8         | 0.7613    | 6.392     | 374.6     | 0.2595    |
# | 9         | 0.8006    | 3.62      | 340.0     | 0.02074   |
# | 10        | 0.8028    | 3.748     | 340.2     | 0.03276   |
# | 11        | 0.7634    | 7.093     | 458.3     | 0.1372    |
# | 12        | 0.7823    | 3.808     | 447.7     | 0.08488   |
# | 13        | 0.7872    | 4.284     | 340.6     | 0.1304    |
# | 14        | 0.7574    | 3.757     | 340.0     | 0.2753    |
# | 15        | 0.8025    | 7.65      | 147.7     | 0.09127   |
# | 16        | 0.7904    | 7.609     | 147.8     | 0.188     |
# | 17        | 0.7911    | 7.78      | 147.8     | 0.1765    |
# | 18        | 0.7959    | 3.471     | 340.4     | 0.04825   |
# | 19        | 0.7826    | 6.045     | 370.7     | 0.174     |
# | 20        | 0.7831    | 7.462     | 147.2     | 0.2471    |
# =============================================================
# Best CatBoost parameters: {'depth': 3, 'iterations': 340, 'learning_rate': 0.032761244104905135}
# Best ensemble weights: XGB: 0.64, CatBoost+PCA: 0.36
# Checkpoints saved to titanic/checkpoints\checkpoint_xgb.pth and titanic/checkpoints\checkpoint_cat.pth
# Loaded checkpoints from files.
# Submission saved to titanic_submission.csv!
