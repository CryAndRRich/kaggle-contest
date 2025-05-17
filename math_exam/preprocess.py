import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

class Data():
    def __init__(self, data_path: str) -> None:
        """
        Initialize Data object by loading raw train and test CSVs, concatenating them,
        and performing initial missing-value handling

        Parameters:
            data_path: Path to the directory containing "train.csv" and "test.csv"
        """
        self.data_dir = os.path.join(os.getcwd(), data_path)

        # Load training and testing data
        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        # Combine data for unified preprocessing
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        self.data["mean_exam_points"] = self.data["mean_exam_points"].fillna(0)
        for col in self.data.select_dtypes(include=["int64", "float64"]).columns:
            self.data[col] = self.data[col].astype("float64")

    def __drop_uninformative(self) -> None:
        """
        Remove features with zero mutual information relative to the target
        Keeps only columns whose mutual information > 0 plus "Id" and "mean_exam_points"
        """
        # Drop known uninformative columns before MI calculation
        self.data = self.data.drop(columns=["age_bin", "mean_price_by_age_bin", "month_of_experience",
                                            "years_of_experience", "mean_price_by_experience", 
                                            "mean_price_by_qualification",
                                            "mean_price_by_subject_binary", "total_subject_binary"])
        mi_scores = self.mi_scores_dataset()
        # Select columns with MI greater than zero
        cols_to_keep = mi_scores[mi_scores > 0.0].index.tolist()
        cols_to_keep += ["Id", "mean_exam_points"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        """
        Compute mutual information scores for all features against "mean_exam_points"
        Saves the MI scores to "mi_score.csv" in the data directory

        Returns:
            pd.Series: MI scores indexed by feature name
        """
        # Use only training subset (Id <= 9999)
        df = self.data[self.data["Id"] <= 9999].copy()
        y = df.pop("mean_exam_points")  # target variable
        df.pop("Id")  # drop identifier

        # Convert categorical to integer codes for MI calculation
        for colname in df.select_dtypes(["object", "category"]):
            df[colname], _ = df[colname].factorize()

        # Determine which features are discrete
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df.dtypes]
        mi = mutual_info_classif(df, y, discrete_features=discrete_features, random_state=0)

        mi_scores = pd.Series(mi, index=df.columns, name="MI Scores").sort_values(ascending=False)
        # mi_scores.to_csv(os.path.join(self.data_dir, "mi_score.csv"), index=True)
        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        """
        Evaluate baseline model performance on the training set using cross-validation
        Reports mean squared error (MSE) and coefficient of determination (R²)

        Parameters:
            model: The model to evaluate. Defaults to XGBRegressor()
        """
        df = self.data[self.data["Id"] <= 9999].copy()
        y = df.pop("mean_exam_points")
        df.pop("Id")

        scoring = {
            "MSE": "neg_mean_squared_error",
            "R²":  "r2"
        }

        # Compute and print each score
        for name, scorer in scoring.items():
            score = cross_val_score(model, df, y, cv=5, scoring=scorer).mean()
            if name == "MSE":
                score = -score
            print(f"Baseline {name}: {score:.5f}")

    def __age_processed(self) -> None:
        """
        Generate age-related features including logarithmic transformations and age bins
        """
        self.data["log_age"] = np.log(self.data["age"])

        # Adjust years of experience and compute month equivalent
        self.data["years_of_experience"] = self.data["years_of_experience"].apply(lambda x: x if x >= 1.0 else 0.5)
        self.data["month_of_experience"] = self.data["years_of_experience"] * 12
        self.data["log_years_of_experience"] = np.log1p(self.data["years_of_experience"])
        self.data["log_month_of_experience"] = np.log(self.data["month_of_experience"])

        # Compute start year as tutor
        self.data["start_tutor"] = self.data["age"] - self.data["years_of_experience"]

        def age_bin(x):
            if x <= 20:
                return 1
            elif 20 < x <= 30:
                return 2
            elif 30 < x <= 40:
                return 3
            elif 40 < x <= 50:
                return 4
            elif 50 < x <= 60:
                return 5
            return 6

        self.data["age_bin"] = self.data["age"].apply(age_bin)

    def __price_processed(self) -> None:
        """
        Create price-related features including squared price and mean price ratios
        """
        self.data["square_price"] = np.square(self.data["lesson_price"])

        # Compute group means and relative price ratios
        self.data["mean_price_by_age_bin"] = self.data.groupby("age_bin", observed=False)["lesson_price"].transform("mean")
        self.data["mean_price_by_experience"] = self.data.groupby("years_of_experience", observed=False)["lesson_price"].transform("mean")
        self.data["mean_price_by_qualification"] = self.data.groupby("qualification", observed=False)["lesson_price"].transform("mean")
        
        self.data["lesson_age"] = self.data["lesson_price"] / self.data["mean_price_by_age_bin"]
        self.data["lesson_experience"] = self.data["lesson_price"] / self.data["mean_price_by_experience"]
        self.data["lesson_qualification"] = self.data["lesson_price"] / self.data["mean_price_by_qualification"]

    def __subject_processed(self) -> None:
        """
        Generate subject-related features including total subject count and binary encodings
        """
        # Sum of subject enrollments
        self.data["total_subject"] = (self.data["physics"] + self.data["chemistry"] + 
                                      self.data["biology"] + self.data["english"] + 
                                      self.data["geography"] + self.data["history"])
        
        # Binary encoding for subject combination
        self.data["total_subject_binary"] = (self.data["physics"] * 2 ** 5 + self.data["chemistry"] * 2 ** 4 + 
                                             self.data["biology"] * 2 ** 3 + self.data["english"] * 2 ** 2 + 
                                             self.data["geography"] * 2 ** 1 + self.data["history"])
        
        self.data["mean_price_by_subject_binary"] = self.data.groupby("total_subject_binary", observed=False)["lesson_price"].transform("mean")
        self.data["lesson_subject_binary"] = self.data["lesson_price"] / self.data["mean_price_by_subject_binary"]

    def __k_means(self) -> None:
        """
        Perform KMeans clustering on selected features to add cluster labels
        and distances to each cluster centroid
        """
        # Features used for clustering
        cluster_features = ["lesson_subject_binary", "lesson_experience", "square_price", 
                            "lesson_qualification", "lesson_age"]
        
        # Standardize features
        X = self.data.loc[:, cluster_features]
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
        
        kmeans = KMeans(n_clusters=10, n_init=50, random_state=0)
        self.data["cluster"] = kmeans.fit_predict(X_scaled)

    def data_processed(self) -> None:
        """
        Execute all preprocessing steps in order:
        age, price, subject, clustering, and feature selection
        """
        self.__age_processed()
        self.__price_processed()
        self.__subject_processed()
        self.__k_means()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        """
        Save the processed data to a CSV file in the data directory

        Returns:
            str: Filename of the saved CSV
        """
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"


if __name__ == "__main__":
    data_path = "math_exam/data"
    data = Data(data_path)

    data.data_processed()
    data.save_csv()
    data.baseline_score_dataset()

# Data saved to processed_data.csv!
# Baseline MSE: 44.67064
# Baseline R²: 0.75616