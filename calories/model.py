# Updated on May 31, 2025, 6:32 PM
# Public Score: 0.05777
# Rank: 1476/4183

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

from preprocess import Data

class CaloriesModel():
    """
    Calories prediction using XGBRegressor with Bayesian hyperparameter tuning on RMSE
    """
    def __init__(self,
                 data_path: str,
                 data_processed_path: str) -> None:
        # Setup paths and load data
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        # Split by id
        self.train_data = data[(data["id"] < 750000) | (data["id"] > 5000000)].copy()
        self.test_data = data[(data["id"] >= 750000) & (data["id"] < 5000000)].copy()

        self.test_ids = self.test_data["id"].astype(int).values
        self.X_train = self.train_data.drop(columns=["Calories", "id"])
        self.y_train = self.train_data["Calories"]
        self.X_test = self.test_data.drop(columns=["Calories", "id"])

        # Identify feature types
        self.numerical_features = (
            self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        )
        self.categorical_features = (
            self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        )

        # Build preprocessing + hyperparameter tuning
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        # Preprocessing
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        self.preprocessor = ColumnTransformer([
            ("num", num_transformer, self.numerical_features),
            ("cat", cat_transformer, self.categorical_features)
        ])

        # Bayesian optimization on ROC-AUC
        self._bayes_opt()

        # Final pipeline with best params
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("xgb", XGBRegressor(
                **self.best_params_xgb,
                random_state=42,
                tree_method="auto"
            ))
        ])

    def _bayes_opt(self) -> None:
        cv = KFold(n_splits=7, shuffle=True, random_state=42)
        
        def xgb_cv(learning_rate: float,
                   max_depth: float,
                   subsample: float,
                   colsample_bytree: float,
                   gamma: float) -> float:
            md = int(max_depth)
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=learning_rate,
                max_depth=md,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                n_jobs= -1
            )
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("xgb", model)
            ])
        
            neg_mse = cross_val_score(pipe, self.X_train, self.y_train, cv=cv,
                                scoring="neg_mean_squared_error").mean()
            rmse = np.sqrt(-neg_mse)
            return -rmse

        pbounds = {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0.0, 5.0)
        }

        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=15)

        params = optimizer.max["params"]
        # Cast integer params
        params["max_depth"] = int(params["max_depth"])
        self.best_params_xgb = params
        print("Best XGB params:", self.best_params_xgb)

    def train(self) -> None:
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self) -> None:
        preds = self.pipeline.predict(self.X_test)

        self.submission = pd.DataFrame({
            "id": self.test_ids,
            "Calories": np.expm1(preds)
        })
        outfile = os.path.join(self.data_dir, "calories_submission.csv")
        self.submission.to_csv(outfile, index=False)
        print("Submission saved to calories_submission.csv!")

if __name__ == "__main__":
    data_path = "calories/data"
    data = Data(data_path)
    data_processed_path = data.save_csv()

    model = CaloriesModel(data_path, data_processed_path)
    model.train()
    model.predict()

# |   iter    |  target   | colsam... |   gamma   | learni... | max_depth | subsample |
# -------------------------------------------------------------------------------------
# | 1         | -0.06511  | 0.6873    | 4.754     | 0.2223    | 7.191     | 0.578     |
# | 2         | -0.06211  | 0.578     | 0.2904    | 0.2612    | 7.208     | 0.854     |
# | 3         | -0.07256  | 0.5103    | 4.85      | 0.2514    | 4.486     | 0.5909    |
# | 4         | -0.06331  | 0.5917    | 1.521     | 0.1622    | 6.024     | 0.6456    |
# | 5         | -0.06254  | 0.8059    | 0.6975    | 0.09472   | 5.565     | 0.728     |
# | 6         | -0.06302  | 0.9506    | 2.549     | 0.2015    | 9.984     | 0.5735    |
# | 7         | -0.05975  | 0.5105    | 0.04016   | 0.07558   | 9.954     | 0.5363    |
# | 8         | -0.2001   | 0.9816    | 0.0152    | 0.01826   | 3.004     | 0.6563    |
# | 9         | -0.06288  | 0.7863    | 4.974     | 0.1134    | 9.859     | 0.9187    |
# | 10        | -0.06185  | 0.99      | 2.452     | 0.1344    | 8.006     | 0.9914    |
# | 11        | -0.06885  | 0.9032    | 4.782     | 0.07638   | 3.031     | 0.9195    |
# | 12        | -0.06378  | 0.516     | 3.862     | 0.08123   | 8.716     | 0.5119    |
# | 13        | -0.07178  | 0.9807    | 3.434     | 0.03454   | 6.129     | 0.959     |
# | 14        | -0.06185  | 0.9471    | 0.8008    | 0.2619    | 8.926     | 0.9681    |
# | 15        | -0.06298  | 0.9551    | 4.983     | 0.1178    | 8.432     | 0.9124    |
# | 16        | -0.06185  | 0.9939    | 0.2087    | 0.05219   | 6.311     | 0.5765    |
# | 17        | -0.06087  | 0.6029    | 1.44      | 0.06771   | 9.975     | 0.9948    |
# | 18        | -0.06209  | 0.5095    | 1.501     | 0.09618   | 8.146     | 0.5114    |
# | 19        | -0.06418  | 0.5022    | 3.859     | 0.2977    | 9.825     | 0.9657    |
# | 20        | -0.08076  | 0.9829    | 0.9364    | 0.0302    | 6.637     | 0.9831    |
# =====================================================================================
# Best XGB params: {"colsample_bytree": 0.5104847748776576, "gamma": 0.04016247992372657, "learning_rate": 0.07558437069278022, "max_depth": 9, "subsample": 0.5362662173001866}
# Submission saved to calories_submission.csv!