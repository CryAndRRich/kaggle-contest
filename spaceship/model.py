# Updated on March 23, 2025, 9:02 AM
# Public Score: 0.80757
# Rank: 88/1834

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

from preprocess import Data

class SpaceshipModel:
    def __init__(self, datasets_path: str, data_processed_path: str) -> None:
        """
        Initialize the SpaceshipModel by loading processed data and preparing training and test sets.
        
        Parameters:
            datasets_path: The relative path to the datasets directory.
            data_processed_path: The filename of the processed data CSV.
        """
        # Set the dataset directory.
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)
        data_path = os.path.join(self.datasets_dir, data_processed_path)
        
        # Read the processed CSV data.
        data = pd.read_csv(data_path)

        # Split data into training and test sets based on the Transported column.
        self.train_data = data[data["Transported"] != -1].copy()
        self.test_data = data[data["Transported"] == -1].copy()
        
        self.test_passenger_ids = self.test_data["PassengerId"]

        self.X_train = self.train_data.drop(columns=["Transported", "PassengerId"])
        self.y_train = self.train_data["Transported"]

        self.X_test = self.test_data.drop(columns=["Transported", "PassengerId"])

        # Identify categorical columns.
        self.categorical_cols = self.X_train.select_dtypes(include=["object"]).columns.tolist()
        self.categorical_cols += ['HomePlanet', 'Destination']

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Build the preprocessing and modeling pipeline, and set up the parameter grid for hyperparameter tuning.
        
        This method creates a ColumnTransformer for one-hot encoding categorical features and then builds a Pipeline
        that includes the preprocessor and a CatBoostClassifier. It also defines the parameter grid for GridSearchCV.
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
            ],
            remainder="passthrough"
        )

        self.pipe = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", CatBoostClassifier(
                iterations=100,
                random_seed=42,
                eval_metric="Logloss",
                verbose=0
            ))
        ])

        self.param_grid = {
            "classifier__depth": [3, 5, 7, 9],
            "classifier__learning_rate": [0.01, 0.1, 0.2, 0.3],
            "classifier__l2_leaf_reg": [1, 5, 10],
            "classifier__bagging_temperature": [0, 0.5, 1]
        }

        self.grid_search = GridSearchCV(
            estimator=self.pipe,
            param_grid=self.param_grid,
            cv=3,           
            scoring="accuracy",
            n_jobs=-1,      
            verbose=2
        )

    def train(self) -> None:
        """
        Train the model by performing a train/validation split and hyperparameter tuning using GridSearchCV.
        
        The method splits the training data, fits GridSearchCV to identify the best hyperparameters,
        prints the best parameters and corresponding accuracy, and evaluates the best model on a validation set.
        """
        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        self.grid_search.fit(X_train_train, y_train_train)

        print("Best parameters found:", self.grid_search.best_params_)
        print("Best CV accuracy:", self.grid_search.best_score_)

        self.best_pipe = self.grid_search.best_estimator_
        preds_valid = self.best_pipe.predict(X_train_test)
        val_acc = accuracy_score(y_train_test, preds_valid)
        print("Validation accuracy:", val_acc)

    def predict(self) -> None:
        """
        Generate predictions on the test dataset using the best found hyperparameters.
        
        This method builds a final pipeline using the best hyperparameters from GridSearchCV, fits the pipeline on the entire 
        training set, predicts the 'Transported' status for the test set, converts predictions to boolean, and saves the results 
        as a CSV submission file.
        """
        final_pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", CatBoostClassifier(
                iterations=100,
                random_seed=42,
                eval_metric="Logloss",
                verbose=0,
                depth=self.grid_search.best_params_["classifier__depth"],
                learning_rate=self.grid_search.best_params_["classifier__learning_rate"],
                l2_leaf_reg=self.grid_search.best_params_["classifier__l2_leaf_reg"],
                bagging_temperature=self.grid_search.best_params_["classifier__bagging_temperature"]
            ))
        ])

        final_pipeline.fit(self.X_train, self.y_train)

        predictions = final_pipeline.predict(self.X_test)
        predictions_bool = predictions.astype(bool)

        submission = pd.DataFrame({
            "PassengerId": self.test_passenger_ids,
            "Transported": predictions_bool
        })
        
        # Save the submission CSV file.
        output_file = os.path.join(self.datasets_dir, "spaceship_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to spaceship_submission.csv!")

if __name__ == "__main__":
    datasets_path = "spaceship/datasets"

    data = Data(datasets_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = SpaceshipModel(datasets_path, data_processed_path)
    model.train()
    model.predict()

#Fitting 3 folds for each of 144 candidates, totalling 432 fits
#[CV] END classifier__bagging_temperature=0, classifier__depth=3, classifier__l2_leaf_reg=1, classifier__learning_rate=0.01; total time=   6.5s
#[CV] END classifier__bagging_temperature=0, classifier__depth=3, classifier__l2_leaf_reg=1, classifier__learning_rate=0.01; total time=   6.6s
#[CV] END classifier__bagging_temperature=0, classifier__depth=3, classifier__l2_leaf_reg=1, classifier__learning_rate=0.1; total time=   6.7s
#[CV] END classifier__bagging_temperature=0, classifier__depth=3, classifier__l2_leaf_reg=1, classifier__learning_rate=0.01; total time=   6.9s
#...
#[CV] END classifier__bagging_temperature=1, classifier__depth=9, classifier__l2_leaf_reg=10, classifier__learning_rate=0.2; total time=  60.0s
#[CV] END classifier__bagging_temperature=1, classifier__depth=9, classifier__l2_leaf_reg=10, classifier__learning_rate=0.3; total time=  59.5s
#[CV] END classifier__bagging_temperature=1, classifier__depth=9, classifier__l2_leaf_reg=10, classifier__learning_rate=0.3; total time=  58.1s
#[CV] END classifier__bagging_temperature=1, classifier__depth=9, classifier__l2_leaf_reg=10, classifier__learning_rate=0.3; total time=  55.7s
#Best parameters found: {'classifier__bagging_temperature': 0, 'classifier__depth': 7, 'classifier__l2_leaf_reg': 1, 'classifier__learning_rate': 0.1}
#Best CV accuracy: 0.8062985332182917
#Validation accuracy: 0.8033352501437608
#Submission saved to spaceship_submission.csv!