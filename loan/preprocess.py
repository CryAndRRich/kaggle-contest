import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

class Data:
    def __init__(self, datasets_path: str) -> None:
        """
        Initialize the Data object by loading raw train and test CSVs,
        concatenating them, and performing initial missing-value handling

        Parameters:
            datasets_path: Path to the directory containing "train.csv" and "test.csv"
        """
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)

        # Load training and testing data
        train_file_path = os.path.join(self.datasets_dir, "train.csv")
        test_file_path = os.path.join(self.datasets_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        # Combine datasets for unified preprocessing
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        # Initial fill for categorical and numerical columns
        categorical_cols = ["Years in current job", "Purpose", "Term"]
        numerical_cols = ["Months since last delinquent", "Bankruptcies", "Credit Default"]

        for col in categorical_cols:
            self.data[col] = self.data[col].fillna("None").astype("category")
        for col in numerical_cols:
            self.data[col] = self.data[col].fillna(0)

    def __drop_uninformative(self) -> None:
        """
        Drop features with zero mutual information relative to the target.
        Keeps only columns whose MI > 0, plus "Id" and "Credit Default"
        """
        mi_scores = self.mi_scores_dataset()
        # Select columns with MI greater than zero
        cols_to_keep = mi_scores[mi_scores > 0.0].index.tolist()
        cols_to_keep += ["Id", "Credit Default"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        """
        Compute mutual information scores for all features against "Credit Default".
        Saves the MI scores to "mi_score.csv" in the datasets directory

        Returns:
            pd.Series: A pandas Series of MI scores, indexed by feature name
        """
        # Use only training subset (Id <= 7499)
        df = self.data[self.data["Id"] <= 7499].copy()
        y = df.pop("Credit Default")  # target variable
        df.pop("Id")  # drop identifier

        # Convert categorical to integer codes for MI calculation
        for colname in df.select_dtypes(["object", "category"]):
            df[colname], _ = df[colname].factorize()

        # Determine which features are discrete
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df.dtypes]
        mi = mutual_info_classif(df, y, discrete_features=discrete_features, random_state=0)

        mi_scores = pd.Series(mi, index=df.columns, name="MI Scores").sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBClassifier = XGBClassifier()) -> None:
        """
        Evaluate baseline model performance on the training set using cross-validation
        Reports Accuracy, Precision, Recall, and F1

        Parameters:
            model: A scikit-learn-compatible classifier
        """
        df = self.data[self.data["Id"] <= 7499].copy()
        y = df.pop("Credit Default")
        df.pop("Id")

        # Encode categorical codes for modeling
        for col in df.select_dtypes(["category"]):
            df[col] = df[col].cat.codes

        # Compute and print metrics
        metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in metrics:
            score = cross_val_score(model, df, y, cv=5, scoring=metric).mean()
            print(f"Baseline {metric.capitalize()}: {score:.5f}")

    def __home_processed(self) -> None:
        """
        Generate features related to home ownership
        """
        def filter_home(val):
            if "Rent" in val:
                return "Rent"
            if "Mortgage" in val:
                return "Mortgage"
            return "Home"

        # Map raw strings to standardized categories
        self.data["Home Ownership"] = (
            self.data["Home Ownership"].apply(filter_home).astype("category")
        )
        # Binary indicator for home ownership
        self.data["hasHome"] = (self.data["Home Ownership"] == "Home").astype(int)

    def __job_processed(self) -> None:
        """
        Engineer income and employment-related features
        """
        # Impute missing Annual Income within each home-ownership group
        self.data["Annual Income"] = (
            self.data.groupby("Home Ownership", observed=False)["Annual Income"]
                .transform(lambda x: x.fillna(x.median()))
        )
        self.data["Monthly Income"] = self.data["Annual Income"] / 12.0

        # Quartile-based income groups (Low, Medium, High, Very High)
        self.data["Income Group"] = pd.qcut(
            self.data["Annual Income"], q=4,
            labels=["Low", "Medium", "High", "Very High"]
        )

        # Log transforms to reduce skew
        self.data["Log Annual Income"] = np.log1p(self.data["Annual Income"])
        self.data["Log Monthly Income"] = np.log1p(self.data["Monthly Income"])
        self.data["Log Monthly Debt"] = np.log1p(self.data["Monthly Debt"])

        # Net income after debt and debt-to-income ratio
        self.data["Income after Debt"] = self.data["Monthly Income"] - self.data["Monthly Debt"]
        self.data["Dept Income Ratio"] = self.data["Monthly Debt"] / self.data["Monthly Income"]

        # Parse "Years in current job" strings (<1, 10+, None) into numeric
        def filter_year(val):
            tok = str(val).split()[0]
            if tok == "<":
                return 0.5
            if tok == "10+":
                return 10.5
            if tok == "None":
                return np.nan
            return int(tok)

        self.data["Years in current job"] = (
            self.data["Years in current job"].apply(filter_year)
        )
        # Impute missing job years by group median
        self.data["Years in current job"] = (
            self.data.groupby("Home Ownership", observed=False)["Years in current job"]
                .transform(lambda x: x.fillna(x.median()))
        )

    def __credit_processed(self) -> None:
        """
        Build credit-related features
        """
        # Binary flags for negative credit events
        self.data["Has Liens"] = (self.data["Tax Liens"] > 0).astype(int)
        self.data["Has Problems"] = (self.data["Number of Credit Problems"] > 0).astype(int)
        self.data["Bankrupted"] = (self.data["Bankruptcies"] > 0).astype(int)

        # Log-transform skewed credit metrics
        self.data["Log Maximum Open Credit"] = np.log1p(self.data["Maximum Open Credit"])
        self.data["Log Current Loan Amount"] = np.log1p(self.data["Current Loan Amount"])
        self.data["Log Current Credit Balance"] = np.log1p(self.data["Current Credit Balance"])

        # Delinquency recency and history length
        self.data["Year since last delinquent"] = self.data["Months since last delinquent"] / 12.0
        self.data["No delinquent"] = (self.data["Months since last delinquent"] == 0).astype(int)

        self.data["Months of Credit History"] = (self.data["Years of Credit History"] * 12)
        self.data["Credit History Group"] = pd.cut(
            self.data["Years of Credit History"],
            bins=[0, 5, 10, 20, 50],
            labels=["Very Short", "Short", "Medium", "Long"]
        )

        # Fill missing credit score by home group median
        self.data["Credit Score"] = (
            self.data.groupby("Home Ownership", observed=False)["Credit Score"]
                .transform(lambda x: x.fillna(x.median()))
        )
        self.data["Log Credit Score"] = np.log1p(self.data["Credit Score"])

        # Interaction: debt ratio multiplied by credit score
        self.data["Debt Score Interaction"] = (self.data["Dept Income Ratio"] * self.data["Credit Score"])

        # Quartile-based credit score group labels
        self.data["Credit Score Group"] = pd.qcut(
            self.data["Credit Score"], q=4,
            labels=["Bad", "Average", "Good", "Excellent"]
        )

    def __k_means(self) -> None:
        """
        Perform KMeans clustering on selected log/interaction features,
        then store cluster label and distances to each centroid.
        """
        cluster_features = [
            "Log Monthly Income", "Log Credit Score",
            "Debt Score Interaction", "Log Current Loan Amount",
            "Log Maximum Open Credit"
        ]
        # Standardize features for clustering
        X_scaled = self.data[cluster_features]
        X_scaled = (X_scaled - X_scaled.mean()) / X_scaled.std()

        kmeans = KMeans(n_clusters=10, n_init=50, random_state=0)
        self.data["Cluster"] = kmeans.fit_predict(X_scaled)

        distances = kmeans.transform(X_scaled)
        # Append distances to centroids as features
        dist_cols = [f"Centroid{i}" for i in range(distances.shape[1])]
        dist_df = pd.DataFrame(distances, columns=dist_cols, index=self.data.index)
        self.data = self.data.join(dist_df)

    def data_processed(self) -> None:
        """
        Execute all preprocessing steps
        """
        self.__home_processed()
        self.__job_processed()
        self.__credit_processed()
        self.__k_means()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        """
        Save the processed data to a CSV file in the datasets directory

        Returns:
            str: The filename of the saved CSV
        """
        output_file = os.path.join(self.datasets_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"


if __name__ == "__main__":
    datasets_path = "loan/datasets"
    data = Data(datasets_path)

    data.data_processed()
    data.mi_scores_dataset()
    data.baseline_score_dataset()
    data.save_csv()

#Baseline Accuracy: 0.75813
#Baseline Precision: 0.61652
#Baseline Recall: 0.37386
#Baseline F1: 0.46526
#Data saved to processed_data.csv!
