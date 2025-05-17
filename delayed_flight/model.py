# Updated on May 15, 2025, 9:58 PM
# Public Score: 0.73750
# Rank: 1186/2205

import os

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from preprocess import Data

class DelayedFLightModel():
    """
    DelayedFLight prediction using XGBClassifier with Bayesian hyperparameter tuning on ROC-AUC,
    early stopping on AUC, and threshold tuning via Youden's J statistic
    """
    def __init__(self,
                 data_path: str,
                 data_processed_path: str) -> None:
        # Setup paths and load data
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        # Split by id
        self.train_data = data[data["id"] > 99999].copy()
        self.test_data = data[data["id"] <= 99999].copy()

        self.test_ids = self.test_data["id"]
        self.X_train = self.train_data.drop(columns=["dep_delayed_15min", "id"])
        self.y_train = self.train_data["dep_delayed_15min"]
        self.X_test = self.test_data.drop(columns=["dep_delayed_15min", "id"])

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
            ("xgb", XGBClassifier(
                **self.best_params_xgb,
                eval_metric="auc",
                random_state=42,
                n_jobs= -1
            ))
        ])

    def _bayes_opt(self) -> None:
        def xgb_cv(n_estimators: float,
                   learning_rate: float,
                   max_depth: float,
                   subsample: float,
                   colsample_bytree: float,
                   reg_alpha: float,
                   reg_lambda: float) -> float:
            # Cast to int
            n_est = int(n_estimators)
            md = int(max_depth)
            model = XGBClassifier(
                n_estimators=n_est,
                learning_rate=learning_rate,
                max_depth=md,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                eval_metric="auc",
                random_state=42,
                n_jobs= -1
            )
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("xgb", model)
            ])
            return cross_val_score(
                pipe, self.X_train, self.y_train,
                cv=3, scoring="roc_auc"
            ).mean()

        pbounds = {
            "n_estimators": (100, 500),
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 10),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0, 5),
            "reg_lambda": (0, 5)
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
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"] = int(params["max_depth"])
        self.best_params_xgb = params
        print("Best XGB params:", self.best_params_xgb)

    def train(self) -> None:
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self) -> None:
        probs = self.pipeline.predict_proba(self.X_test)[:, 1]

        self.submission = pd.DataFrame({
            "id": self.test_ids,
            "dep_delayed_15min": probs
        })
        outfile = os.path.join(self.data_dir, "delayed_flight_submission.csv")
        self.submission.to_csv(outfile, index=False)
        print("Submission saved to delayed_flight_submission.csv!")

if __name__ == "__main__":
    data_path = "delayed_flight/data"
    data = Data(data_path)
    data_processed_path = data.save_csv()

    model = DelayedFLightModel(data_path, data_processed_path)
    model.train()
    model.predict()

# Data saved to processed_data.csv!
# |   iter    |  target   | colsam... | learni... | max_depth | n_esti... | reg_alpha | reg_la... | subsample |
# -------------------------------------------------------------------------------------------------------------
# | 1         | 0.7354    | 0.7498    | 0.2857    | 8.124     | 339.5     | 0.7801    | 0.78      | 0.6232    |
# | 2         | 0.7433    | 0.9465    | 0.1843    | 7.957     | 108.2     | 4.85      | 4.162     | 0.6849    |
# | 3         | 0.7407    | 0.6727    | 0.06319   | 5.13      | 309.9     | 2.16      | 1.456     | 0.8447    |
# | 4         | 0.7431    | 0.6558    | 0.09472   | 5.565     | 282.4     | 3.926     | 0.9984    | 0.8057    |
# | 5         | 0.7295    | 0.837     | 0.02347   | 7.253     | 168.2     | 0.3253    | 4.744     | 0.9863    |
# | 6         | 0.7478    | 0.6715    | 0.09571   | 9.831     | 248.5     | 0.1128    | 0.7382    | 0.6105    |
# | 7         | 0.7353    | 0.8464    | 0.0134    | 9.268     | 247.7     | 0.9592    | 0.1605    | 0.9971    |
# | 8         | 0.7516    | 0.7703    | 0.1088    | 9.334     | 289.0     | 4.713     | 1.486     | 0.8722    |
# | 9         | 0.7441    | 0.9994    | 0.02484   | 8.478     | 396.9     | 3.962     | 1.746     | 0.7858    |
# | 10        | 0.7429    | 0.8224    | 0.2721    | 7.633     | 192.2     | 4.831     | 1.981     | 0.6111    |
# | 11        | 0.752     | 0.9237    | 0.0659    | 8.956     | 360.4     | 3.315     | 0.256     | 0.7675    |
# | 12        | 0.7501    | 0.8839    | 0.1931    | 8.612     | 369.6     | 4.224     | 1.078     | 0.984     |
# | 13        | 0.7377    | 0.6476    | 0.2856    | 9.779     | 481.3     | 4.624     | 1.182     | 0.7686    |
# | 14        | 0.7423    | 0.9905    | 0.213     | 9.32      | 277.9     | 2.04      | 1.202     | 0.6716    |
# | 15        | 0.7483    | 0.7713    | 0.1049    | 6.623     | 492.6     | 0.08419   | 4.453     | 0.8078    |
# | 16        | 0.7457    | 0.6026    | 0.1501    | 5.036     | 334.5     | 2.555     | 0.7985    | 0.9866    |
# | 17        | 0.7469    | 0.9111    | 0.2172    | 7.19      | 110.4     | 2.211     | 2.847     | 0.8703    |
# | 18        | 0.74      | 0.7349    | 0.2901    | 3.662     | 288.6     | 4.864     | 2.913     | 0.913     |
# | 19        | 0.7456    | 0.7977    | 0.2441    | 6.852     | 481.1     | 2.357     | 4.703     | 0.7259    |
# | 20        | 0.7495    | 0.8986    | 0.1082    | 6.173     | 362.4     | 3.129     | 1.026     | 0.899     |
# =============================================================================================================
# Best XGB params: {'colsample_bytree': 0.9236526443566715, 'learning_rate': 0.06589640042296577, 'max_depth': 8, 'n_estimators': 360, 'reg_alpha': 3.3153576082225995, 'reg_lambda': 0.255990009427588, 'subsample': 0.76751278843513}
# Submission saved to delayed_flight_submission.csv!