import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

pd.set_option("future.no_silent_downcasting", True)

class Data():
    def __init__(self, data_path: str) -> None:
        """
        Initialize the Data object by loading data
        
        Parameters:
            data_path: Relative path to the data directory
        """
        self.data_dir = os.path.join(os.getcwd(), data_path)
        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        # Concatenate train and test data to work on a unified dataset.
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        # Replace missing values in the "Transported" column with -1 and convert to integer.
        self.data["Transported"] = self.data["Transported"].fillna(-1)
        self.data["Transported"] = self.data["Transported"].astype(int)

        # Map HomePlanet categorical values to numerical representations.
        home_mapping = {
            "Earth": 1,
            "Europa": 2,
            "Mars": 3
        }
        self.data["HomePlanet"] = self.data["HomePlanet"].replace(home_mapping)

        # Map Destination categorical values to numerical representations.
        dest_mapping = {
            "TRAPPIST-1e": 1,
            "PSO J318.5-22": 2,
            "55 Cancri e": 3
        }
        self.data["Destination"] = self.data["Destination"].replace(dest_mapping)

    def mi_scores(self) -> pd.Series:
        """
        Calculate mutual information scores for binary classification
        
        Returns:
            pd.Series: Sorted mutual information scores for each feature
        """
        # Select rows where the target variable is known.
        X = self.data.loc[self.data["Transported"] != -1].copy()
        y = X.pop("Transported")
        
        # Factorize categorical features into numerical codes.
        for colname in X.select_dtypes(include=["object", "category"]).columns:
            X[colname], _ = X[colname].factorize()
        
        # Identify which features are discrete (i.e., integer types).
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        # Compute mutual information scores between features and the target.
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        # Reset index to create a DataFrame and rename columns.
        mi_scores_df = mi_scores.reset_index()
        mi_scores_df.columns = ["Feature", "MI Score"]

        # Save the mutual information scores to a text file.
        output_file = os.path.join(self.data_dir, "mi_scores.txt")
        mi_scores_df.to_csv(output_file, index=False)
        return mi_scores

    def baseline_score(self, model: XGBClassifier = XGBClassifier()) -> None:
        """
        Evaluate and print the baseline performance scores using cross-validation
        
        Parameters:
            model: The model to evaluate. Defaults to XGBClassifier()
        """
        # Select rows where the target variable is known.
        X = self.data.loc[self.data["Transported"] != -1].copy()
        y = X.pop("Transported")
        
        # Factorize categorical features.
        for colname in X.select_dtypes(include=["object", "category"]).columns:
            X[colname], _ = X[colname].factorize()
        
        # Compute cross-validated accuracy and ROC AUC scores.
        accuracy = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
        roc_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
        
        print(f"Baseline Accuracy: {accuracy:.5f}")
        print(f"Baseline ROC AUC: {roc_auc:.5f}")

    def __id_processed(self):
        """
        Process the PassengerId by splitting it into "Group" and "NumInGroup"
        """
        self.data[["Group", "NumInGroup"]] = self.data["PassengerId"].str.split("_", expand=True)

        self.data["Group"] = self.data["Group"].astype(int)
        self.data["NumInGroup"] = self.data["NumInGroup"].astype(int)

    def __location_processed(self):
        """
        Process missing values in location-related features (HomePlanet and Destination) 
        by filling them using group modes or overall mode
        """
        self.data["HomePlanet"] = self.data["HomePlanet"].fillna(
            self.data.groupby("Group")["HomePlanet"].transform(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
        )
        self.data["HomePlanet"] = self.data["HomePlanet"].fillna(self.data["HomePlanet"].mode()[0])
        
        self.data["Destination"] = self.data["Destination"].fillna(
            self.data.groupby("Group")["Destination"].transform(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
        )
        self.data["Destination"] = self.data["Destination"].fillna(self.data["Destination"].mode()[0])

    def __name_processed(self):
        """
        Process the Name column by splitting it into first and last names
        """
        self.data[["FirstName", "LastName"]] = self.data["Name"].str.split(" ", expand=True)
        # Fill missing names with "None" and convert to category type.
        self.data["FirstName"] = self.data["FirstName"].fillna("None").astype("category")
        self.data["LastName"] = self.data["LastName"].fillna("None").astype("category")

    def __cabin_processed(self):
        """
        Process the Cabin column by splitting it into DeckCab, NumCab, and SideCab,
        and then recombine them after filling missing values
        """
        self.data[["DeckCab", "NumCab", "SideCab"]] = self.data["Cabin"].str.split("/", expand=True)

        self.data["DeckCab"] = self.data["DeckCab"].fillna(self.data["DeckCab"].mode()[0])
        self.data["NumCab"] = self.data["NumCab"].fillna(self.data["NumCab"].mode()[0])
        self.data["SideCab"] = self.data["SideCab"].fillna(self.data["SideCab"].mode()[0])

        # Recombine the processed cabin components back into a single string.
        self.data["Cabin"] = self.data["DeckCab"].astype(str) + "/" + self.data["NumCab"].astype(str) + "/" + self.data["SideCab"].astype(str)

    def __age_processed(self):
        """
        Process the Age column by imputing missing values based on group medians or HomePlanet means
        """
        # Compute the median age for each group ignoring missing values.
        group_med = self.data.loc[self.data["Age"].notnull()].groupby("Group", observed=False)["Age"].median()

        def fill_by_group(row):
            if pd.isna(row["Age"]):
                grp = row["Group"]
                # If the group has a median value, fill the missing age with it.
                if grp in group_med.index and not pd.isna(group_med[grp]):
                    return group_med[grp]
            return row["Age"]

        # Fill missing ages using group medians.
        self.data["Age"] = self.data.apply(fill_by_group, axis=1)

        # Compute mean age per HomePlanet for additional imputation.
        homeplanet_med = self.data.loc[self.data["Age"].notnull()].groupby("HomePlanet", observed=False)["Age"].mean()

        def fill_by_homeplanet(row):
            if pd.isna(row["Age"]):
                hp = row["HomePlanet"]
                if hp in homeplanet_med.index:
                    return int(homeplanet_med[hp])
            return row["Age"]

        # Fill remaining missing ages using HomePlanet averages.
        self.data["Age"] = self.data.apply(fill_by_homeplanet, axis=1)

    def __service_processed(self):
        """
        Process the service usage related features by handling missing values
        """
        # Map string values to binary for CryoSleep and VIP.
        self.data["CryoSleep"] = self.data["CryoSleep"].map({"TRUE": 1, "FALSE": 0})
        self.data["VIP"] = self.data["VIP"].map({"TRUE": 1, "FALSE": 0})

        # Create an "Active" flag based on service usage.
        cond1 = (self.data["RoomService"].notnull() & self.data["RoomService"] > 0)
        cond2 = (self.data["FoodCourt"].notnull() & self.data["FoodCourt"] > 0)
        cond3 = (self.data["ShoppingMall"].notnull() & self.data["ShoppingMall"] > 0)
        cond4 = (self.data["Spa"].notnull() & self.data["Spa"] > 0)
        cond5 = (self.data["VRDeck"].notnull() & self.data["VRDeck"] > 0)
        self.data["Active"] = (cond1 | cond2 | cond3 | cond4 | cond5)

        # For active users, fill missing CryoSleep values with 0; for inactive, with 1.
        self.data.loc[self.data["Active"] == True, "CryoSleep"] = self.data.loc[self.data["Active"] == True, "CryoSleep"].fillna(0)
        self.data.loc[self.data["Active"] == False, "CryoSleep"] = self.data.loc[self.data["Active"] == False, "CryoSleep"].fillna(1)
        
        # For passengers in cryosleep, set all service usage values to 0 if missing.
        self.data.loc[self.data["CryoSleep"] == 1, ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] = \
            self.data.loc[self.data["CryoSleep"] == 1, ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0)
        
        # For each service, fill missing values with the median for the corresponding CryoSleep status.
        self.data["RoomService"] = self.data["RoomService"].fillna(self.data.groupby(["CryoSleep"])["RoomService"].transform("median"))
        self.data["FoodCourt"] = self.data["FoodCourt"].fillna(self.data.groupby(["CryoSleep"])["FoodCourt"].transform("median"))
        self.data["ShoppingMall"] = self.data["ShoppingMall"].fillna(self.data.groupby(["CryoSleep"])["ShoppingMall"].transform("median"))
        self.data["Spa"] = self.data["Spa"].fillna(self.data.groupby(["CryoSleep"])["Spa"].transform("median"))
        self.data["VRDeck"] = self.data["VRDeck"].fillna(self.data.groupby(["CryoSleep"])["VRDeck"].transform("median"))

        # Compute the total bill by summing all service charges.
        self.data["TotalBill"] = (self.data["RoomService"] + self.data["FoodCourt"] + 
                                  self.data["ShoppingMall"] + self.data["Spa"] + self.data["VRDeck"])

        # Calculate the average total bill for later imputation of VIP status.
        bill_med = self.data["TotalBill"].mean()

        def fill_vip(row):
            # If VIP status is missing, set it based on whether the TotalBill is above average.
            if pd.isna(row["VIP"]):
                return int(row["TotalBill"] >= bill_med)
            return row["VIP"]
        
        self.data["VIP"] = self.data.apply(fill_vip, axis=1)

        # Create new features by multiplying TotalBill with one-hot encoded Destination columns.
        dest = pd.get_dummies(self.data["Destination"], prefix="Hp").mul(self.data["TotalBill"], axis=0)
        self.data = self.data.join(dest)
    
    def __k_means(self):
        """
        Apply KMeans clustering to selected service-related features and append 
        the cluster labels and distances to the data.
        """
        # Define features to use for clustering.
        cluster_features = ["VRDeck", "Spa", "ShoppingMall", "FoodCourt", "RoomService", "TotalBill"]
        
        # Standardize the features (z-score normalization).
        X_scaled = self.data.loc[:, cluster_features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        
        # Initialize and fit the KMeans model.
        kmeans = KMeans(n_clusters=10, n_init=50, random_state=0)
        self.data["Cluster"] = kmeans.fit_predict(X_scaled)
        
        # Compute distances of each point from each cluster centroid.
        distances = kmeans.transform(X_scaled)
        
        # Create a DataFrame for the distances and append it to the main data.
        distances_df = pd.DataFrame(
            distances, 
            columns=[f"Centroid{i}" for i in range(distances.shape[1])],
            index=self.data.index
        )
        
        self.data = self.data.join(distances_df)
    
    def data_processed(self):
        """
        Apply all processing functions to clean and feature-engineer the dataset.
        """
        self.__id_processed()
        self.__location_processed()
        self.__name_processed()
        self.__cabin_processed()
        self.__age_processed()
        self.__service_processed()
        self.__k_means()
        # Drop columns that are no longer needed.
        self.data = self.data.drop(columns=["Name", "Active"])

    def save_csv(self) -> str:
        """
        Save the processed data to a CSV file
        
        Returns:
            str: The output file name
        """
        self.output = "processed_data.csv"
        output_file = os.path.join(self.data_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    data_path = "spaceship/data"
    data = Data(data_path)
    data.data_processed()
    data.mi_scores()
    data.baseline_score()
    data.save_csv()

# Baseline Accuracy: 0.67584
# Baseline ROC AUC: 0.84504
