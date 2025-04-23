# Updated on March 30, 2025, 05:42 PM
# Public Score: 0.001
# Rank: 20/44

import os
import pandas as pd

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from preprocess import Data

class MolecularModel:
    def __init__(self, datasets_path: str, data_processed_path: str) -> None:
        """
        Initialize the MolecularModel instance

        Params:
            datasets_path: Relative path to the datasets directory
            data_processed_path: Relative path to the processed data CSV file
        """
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)
        data_path = os.path.join(self.datasets_dir, data_processed_path)
        
        data = pd.read_csv(data_path)

        self.train_data = data[data["Batch_ID"].str.contains("Train")].copy()
        self.test_data = data[data["Batch_ID"].str.contains("Test")].copy()
        
        self.test_ids = self.test_data["Batch_ID"]

        self.X_train = self.train_data.drop(columns=["Batch_ID", "T80"])
        self.y_train = self.train_data["T80"]

        self.X_test = self.test_data.drop(columns=["Batch_ID", "T80"])

        self.numerical_features = self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Build the data preprocessing pipelines for numerical and categorical features,
        and initialize the hyperparameters for the XGBRegressor
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
        self.params = {
            "gamma": 1.446443962043955,
            "learning_rate": 0.09157530923918498,
            "max_depth": 10,
            "min_child_weight": 2.19384448681566,
            "n_estimators": 831,
            "random_state": 42
        }

    def predict(self) -> None:
        """
        Generate predictions on the test dataset using the best found hyperparameters.

        This method builds a final pipeline using the preprocessor and XGBRegressor,
        fits the pipeline on the entire training set, predicts the "T80" values for the test set,
        applies a constant offset, and saves the results as a CSV submission file.
        """
        final_pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("regressor", XGBRegressor(**self.params))
        ])

        final_pipeline.fit(self.X_train, self.y_train)

        predictions = final_pipeline.predict(self.X_test)

        surprise_c = -0.1
        submission = pd.DataFrame({
            "Batch_ID": self.test_ids,
            "T80": predictions + surprise_c
        })
        
        # Save the submission CSV file.
        output_file = os.path.join(self.datasets_dir, "molecular_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to molecular_submission.csv!")


if __name__ == "__main__":
    datasets_path = "molecular/datasets"

    data = Data(datasets_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = MolecularModel(datasets_path, data_processed_path)
    model.predict()
