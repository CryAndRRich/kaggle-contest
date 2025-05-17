# Updated on May 17, 2025, 09:50 PM
# Public Score: 0.78406
# Rank: 9/151

import os
import pandas as pd

from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

from preprocess import Data

class MathExamModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        """
        Initialize MathExamModel by loading processed data and splitting into train and test sets

        Parameters:
            data_path: Relative path to the data directory
            data_processed_path: Filename of the processed data CSV inside data_path
        """
        # Set dataset directory and load processed data
        self.data_dir = os.path.join(os.getcwd(), data_path)
        csv_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(csv_path)

        # Split into training and test based on Id
        self.train_data = data[data["Id"] <= 9999].copy()
        self.test_data = data[data["Id"] > 9999].copy()

        # Separate target variable
        self.y_train = self.train_data.pop("mean_exam_points")
        self.train_data.pop("Id")

        # Store test identifiers and drop target from test set
        self.test_ids = self.test_data["Id"].astype(int).values
        self.test_data.drop(columns=["Id", "mean_exam_points"], inplace=True, errors="ignore")

        # Run Bayesian optimization to find best hyperparameters
        self._bayes_opt()

    def train(self) -> None:
        """
        Train the XGBoost regressor using the best hyperparameters found
        """
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=self.best_params["max_depth"],
            learning_rate=self.best_params["learning_rate"],
            subsample=self.best_params["subsample"],
            colsample_bytree=self.best_params["colsample_bytree"],
            gamma=self.best_params["gamma"],
            random_state=42,
            tree_method="auto"
        )
        self.model.fit(self.train_data, self.y_train)

    def _bayes_opt(self) -> None:
        """
        Perform Bayesian optimization on XGBoost hyperparameters

        Defines an inner cross-validation objective that maximizes R² score over
        a 5-fold KFold split, then runs optimizer over specified parameter bounds.
        Sets self.best_params to the best found hyperparameter values.
        """
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def xgb_cv(max_depth: float,
                   learning_rate: float,
                   subsample: float,
                   colsample_bytree: float,
                   gamma: float) -> float:
            """
            Objective for Bayesian optimization: return mean R² via CV

            Parameters:
                max_depth: Maximum tree depth (as float, will be cast to int)
                learning_rate: Boosting learning rate
                subsample: Subsample ratio of the training instances
                colsample_bytree: Subsample ratio of columns when constructing each tree
                gamma: Minimum loss reduction required to make a further partition

            Returns:
                float: Mean R² score across CV folds
            """
            params = {
                "n_estimators": 100,
                "max_depth": int(max_depth),
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "gamma": gamma,
                "random_state": 42,
                "tree_method": "auto"
            }
            model = XGBRegressor(**params)
            score = cross_val_score(model, self.train_data, self.y_train,
                                    cv=cv, scoring="r2").mean()
            return score

        bounds = {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.5, 1),
            "colsample_bytree": (0.5, 1),
            "gamma": (0, 5)
        }

        optimizer = BayesianOptimization(f=xgb_cv, pbounds=bounds, random_state=42)
        optimizer.maximize(init_points=5, n_iter=15)

        best = optimizer.max["params"]
        best["max_depth"] = int(best["max_depth"])
        self.best_params = best
        print("Best XGBoost params:", self.best_params)

    def predict(self) -> None:
        """
        Generate predictions on the test set, apply post-processing shift,
        and save results to CSV

        Post-processing: subtract 0.1 from predictions above 40,
        add 0.1 otherwise
        """
        preds = self.model.predict(self.test_data)
        adjusted = [x - 0.1 if x > 40 else x + 0.1 for x in preds]

        submission = pd.DataFrame({
            "Id": self.test_ids,
            "mean_exam_points": adjusted
        })

        output_file = os.path.join(self.data_dir, "math_exam_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to math_exam_submission.csv!")


if __name__ == "__main__":
    data_path = "math_exam/data"

    data = Data(data_path)
    data.data_processed()
    processed = data.save_csv()

    model = MathExamModel(data_path, processed)
    model.train()
    model.predict()


# Data saved to processed_data.csv!
# |   iter    |  target   | colsam... |   gamma   | learni... | max_depth | subsample |
# -------------------------------------------------------------------------------------
# | 1         | 0.7434    | 0.6873    | 4.754     | 0.2223    | 7.191     | 0.578     |
# | 2         | 0.7437    | 0.578     | 0.2904    | 0.2612    | 7.208     | 0.854     |
# | 3         | 0.7742    | 0.5103    | 4.85      | 0.2514    | 4.486     | 0.5909    |
# | 4         | 0.7703    | 0.5917    | 1.521     | 0.1622    | 6.024     | 0.6456    |
# | 5         | 0.7803    | 0.8059    | 0.6975    | 0.09472   | 5.565     | 0.728     |
# | 6         | 0.7697    | 0.8724    | 1.184     | 0.2235    | 5.017     | 0.7964    |
# | 7         | 0.7785    | 0.6676    | 0.008038  | 0.1132    | 5.119     | 0.5502    |
# | 8         | 0.7781    | 0.6301    | 4.886     | 0.2056    | 3.071     | 0.7581    |
# | 9         | 0.7787    | 0.5067    | 3.611     | 0.2195    | 3.349     | 0.6429    |
# | 10        | 0.7539    | 0.5228    | 2.219     | 0.02842   | 3.007     | 0.7618    |
# | 11        | 0.7759    | 0.966     | 3.918     | 0.2402    | 4.205     | 0.9264    |
# | 12        | 0.7361    | 0.703     | 2.535     | 0.1709    | 9.999     | 0.9161    |
# | 13        | 0.7792    | 0.8536    | 0.02572   | 0.2186    | 3.021     | 0.8411    |
# | 14        | 0.78      | 0.9183    | 0.001909  | 0.09246   | 4.142     | 0.9764    |
# | 15        | 0.7705    | 0.9679    | 4.242     | 0.0454    | 3.084     | 0.5934    |
# | 16        | 0.7782    | 0.5022    | 3.152     | 0.03452   | 5.151     | 0.5497    |
# | 17        | 0.7552    | 0.8958    | 4.993     | 0.2478    | 9.973     | 0.9937    |
# | 18        | 0.6827    | 0.9815    | 0.01903   | 0.2958    | 9.891     | 0.5531    |
# | 19        | 0.6906    | 0.9952    | 0.1008    | 0.01132   | 5.671     | 0.984     |
# | 20        | 0.7795    | 0.9319    | 2.861     | 0.1307    | 4.283     | 0.6407    |
# =====================================================================================
# Best XGBoost params: {'colsample_bytree': 0.8059264473611898, 'gamma': 0.6974693032602092, 'learning_rate': 0.09472194807521325, 'max_depth': 5, 'subsample': 0.728034992108518}
# Submission saved to math_exam_submission.csv!