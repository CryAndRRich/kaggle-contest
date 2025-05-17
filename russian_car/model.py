# Updated on April 5, 2025, 04:52 PM
# Public Score: 38.8435
# Rank: 59/294

import os
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from preprocess import Data

class RussianCarModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        """
        Initialize the RussianCarModel instance

        Params:
            data_path: Relative path to the data directory
            data_processed_path: Relative path to the processed data CSV file
        """
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        
        data = pd.read_csv(data_path)

        self.train_data = data[data["id"] <= 51635].copy()
        self.test_data = data[data["id"] > 51635].copy()
        
        self.test_ids = self.test_data["id"]

        self.X_train = self.train_data.drop(columns=["price", "id"])
        self.y_train = self.train_data["price"]

        self.X_test = self.test_data.drop(columns=["price", "id"])

        self.numerical_features = self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Build the data preprocessing pipelines for numerical and categorical features,
        and initialize the hyperparameters for the XGBRegressor and CatBoost
        """
        self.numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        self.categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.numerical_pipeline, self.numerical_features),
                ("cat", self.categorical_pipeline, self.categorical_features)
            ]
        )

        # Hyperparameters for XGBRegressor fine-tuned by Bayesian optimization
        self.xgb_params = {
            "gamma": 0.05236979007555209,
            "learning_rate": 0.042448977633043686,
            "max_depth": 10,
            "min_child_weight": 3.4507554274940455,
            "n_estimators": 749
        }

        # Hyperparameters for CatBoostRegressor fine-tuned by Bayesian optimization
        self.cat_params = {
            "depth": 9,
            "iterations": 788,
            "l2_leaf_reg": 1.5752515913647136,
            "learning_rate": 0.25021803226206724,
            "verbose": 0
        }

        self.xgb_pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", XGBRegressor(
                random_state=42,
                **self.xgb_params
            ))
        ])
        self.cat_pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", CatBoostRegressor(
                random_state=42,
                **self.cat_params
            ))
        ])

    def train(self) -> None:
        """
        Train the XGBoost and CatBoost pipelines, then find the best ensemble weights based on validation data.
        """
        # Split the data into training and validation sets for ensemble tuning
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

        # Train both pipelines on the training split
        self.xgb_pipeline.fit(X_train_split, y_train_split)
        self.cat_pipeline.fit(X_train_split, y_train_split)

        # Predict probabilities on the validation set for each model
        val_pred_xgb = self.xgb_pipeline.predict_proba(X_val)
        val_pred_cat = self.cat_pipeline.predict_proba(X_val)

        # Find the best ensemble weights (weighted average) based on MAE
        self.best_w_xgb, self.best_w_cat = self.find_best_ensemble_weights(
            y_val, val_pred_xgb, val_pred_cat, step=0.01
        )
        print("Best ensemble weights: XGB: {:.2f}, CatBoost: {:.2f}".format(self.best_w_xgb, self.best_w_cat))

    def find_best_ensemble_weights(self, 
                                   y_true: np.ndarray,
                                   y_pred_xgb: np.ndarray,
                                   y_pred_cat: np.ndarray,
                                   step: float = 0.01) -> Tuple[float, float]:
        """
        Find the best ensemble weights for combining predictions from two models (weighted average)
        based on MAE

        Parameters:
            y_true: Ground truth target values from the validation set
            y_pred_xgb: Predictions from the XGBoost model
            y_pred_cat: Predictions from the CatBoost model
            step: Step size for iterating through weights from 0 to 1. Defaults to 0.01

        Returns:
            Tuple[float, float]: Weights for XGBoost and CatBoost (corresponding to w and 1-w).
        """
        def mae(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))
        
        best_w = 0.0
        best_mae = float("inf")
        for w in np.arange(0, 1 + step, step):
            ensemble_pred = w * y_pred_xgb + (1 - w) * y_pred_cat
            current_mae = mae(y_true, ensemble_pred)
            if current_mae < best_mae:
                best_w = w
                best_mae = current_mae
        return best_w, 1 - best_w

    def predict(self) -> None:
        """
        Predict car prices on the test set using the trained ensemble model

        This method retrains both pipelines on the full training dataset,
        makes predictions on the test data, computes a weighted ensemble of the results,
        exponentiates the predictions (to reverse log-transform), and writes them to a CSV file
        """
        self.xgb_pipeline.fit(self.X_train, self.y_train)
        self.cat_pipeline.fit(self.X_train, self.y_train)

        preds_xgb = self.xgb_pipeline.predict(self.X_test)
        preds_cat = self.cat_pipeline.predict(self.X_test)

        ensemble_preds = self.best_w_xgb * preds_xgb + self.best_w_cat * preds_cat
        surprise_c = -2450
        submission = pd.DataFrame({
            "id": self.test_ids,
            "price": np.exp(ensemble_preds) + surprise_c
        })
        
        # Save the submission CSV file.
        output_file = os.path.join(self.data_dir, "russian_car_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to russian_car_submission.csv!")


if __name__ == "__main__":
    data_path = "russian_car/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = RussianCarModel(data_path, data_processed_path)
    model.predict()
