import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

pd.set_option("future.no_silent_downcasting", True)

class Data:
    def __init__(self, datasets_path: str) -> None:
        """
        Initialize the Data object by loading datasets.
        
        Parameters:
            datasets_path (str): Relative path to the datasets directory.
        """
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)

        # Build file paths for training and testing datasets.
        train_file_path = os.path.join(self.datasets_dir, "train.csv")
        test_file_path = os.path.join(self.datasets_dir, "test.csv")

        # Read CSV files.
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        # Concatenate train and test data.
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        self.data["Transported"] = self.data["Transported"].fillna(-1)
        self.data["Transported"] = self.data["Transported"].astype(int)

        home_mapping = {
            "Earth": 1,
            "Europa": 2,
            "Mars": 3
        }
        self.data["HomePlanet"] = self.data["HomePlanet"].replace(home_mapping)

        dest_mapping = {
            "TRAPPIST-1e": 1,
            "PSO J318.5-22": 2,
            "55 Cancri e": 3
        }
        self.data["Destination"] = self.data["Destination"].replace(dest_mapping)

    def mi_scores(self) -> pd.Series:
        """
        Calculate mutual information scores for binary classification.
        """
        X = self.data.loc[self.data["Transported"] != -1].copy()
        y = X.pop("Transported")
        
        for colname in X.select_dtypes(include=["object", "category"]).columns:
            X[colname], _ = X[colname].factorize()
        
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        mi_scores_df = mi_scores.reset_index()
        mi_scores_df.columns = ["Feature", "MI Score"]

        output_file = os.path.join(self.datasets_dir, "mi_scores.txt")
        mi_scores_df.to_csv(output_file, index=False)

    def baseline_score(self, model: XGBClassifier = XGBClassifier()) -> None:
        X = self.data.loc[self.data["Transported"] != -1].copy()
        y = X.pop("Transported")
        
        for colname in X.select_dtypes(include=["object", "category"]).columns:
            X[colname], _ = X[colname].factorize()
        
        accuracy = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
        roc_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
        
        print(f"Baseline Accuracy: {accuracy:.5f}")
        print(f"Baseline ROC AUC: {roc_auc:.5f}")

    def __id_processed(self):
        self.data[["Group", "NumInGroup"]] = self.data["PassengerId"].str.split("_", expand=True)

        self.data["Group"] = self.data["Group"].astype(int)
        self.data["NumInGroup"] = self.data["NumInGroup"].astype(int)

    def __location_processed(self):
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
        self.data[["FirstName", "LastName"]] = self.data["Name"].str.split(" ", expand=True)
        self.data["FirstName"] = self.data["FirstName"].fillna("None").astype("category")
        self.data["LastName"] = self.data["LastName"].fillna("None").astype("category")

    def __cabin_processed(self):
        self.data[["DeckCab", "NumCab", "SideCab"]] = self.data["Cabin"].str.split("/", expand=True)

        self.data["DeckCab"] = self.data["DeckCab"].fillna(self.data["DeckCab"].mode()[0])
        self.data["NumCab"] = self.data["NumCab"].fillna(self.data["NumCab"].mode()[0])
        self.data["SideCab"] = self.data["SideCab"].fillna(self.data["SideCab"].mode()[0])

        self.data["Cabin"] = self.data["DeckCab"].astype(str) + "/" + self.data["NumCab"].astype(str) + "/" + self.data["SideCab"].astype(str)

    def __age_processed(self):
        group_med = self.data.loc[self.data["Age"].notnull()].groupby("Group", observed=False)["Age"].median()

        def fill_by_group(row):
            if pd.isna(row["Age"]):
                grp = row["Group"]
                if grp in group_med.index and not pd.isna(group_med[grp]):
                    return group_med[grp]
            return row["Age"]

        self.data["Age"] = self.data.apply(fill_by_group, axis=1)

        homeplanet_med = self.data.loc[self.data["Age"].notnull()].groupby("HomePlanet", observed=False)["Age"].mean()

        def fill_by_homeplanet(row):
            if pd.isna(row["Age"]):
                hp = row["HomePlanet"]
                if hp in homeplanet_med.index:
                    return int(homeplanet_med[hp])
            return row["Age"]

        self.data["Age"] = self.data.apply(fill_by_homeplanet, axis=1)

    def __service_processed(self):
        self.data["CryoSleep"] = self.data["CryoSleep"].map({"TRUE": 1, "FALSE": 0})
        self.data["VIP"] = self.data["VIP"].map({"TRUE": 1, "FALSE": 0})

        cond1 = (self.data["RoomService"].notnull() & self.data["RoomService"] > 0)
        cond2 = (self.data["FoodCourt"].notnull() & self.data["FoodCourt"] > 0)
        cond3 = (self.data["ShoppingMall"].notnull() & self.data["ShoppingMall"] > 0)
        cond4 = (self.data["Spa"].notnull() & self.data["Spa"] > 0)
        cond5 = (self.data["VRDeck"].notnull() & self.data["VRDeck"] > 0)
        self.data["Active"] = (cond1 | cond2 | cond3 | cond4 | cond5)

        self.data.loc[self.data["Active"] == True, "CryoSleep"] = self.data.loc[self.data["Active"] == True, "CryoSleep"].fillna(0)
        self.data.loc[self.data["Active"] == False, "CryoSleep"] = self.data.loc[self.data["Active"] == False, "CryoSleep"].fillna(1)
        
        self.data.loc[self.data["CryoSleep"] == 1, ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] = self.data.loc[self.data["CryoSleep"] == 1, ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0)
        
        self.data["RoomService"] = self.data["RoomService"].fillna(self.data.groupby(["CryoSleep"])["RoomService"].transform("median"))
        self.data["FoodCourt"] = self.data["FoodCourt"].fillna(self.data.groupby(["CryoSleep"])["FoodCourt"].transform("median"))
        self.data["ShoppingMall"] = self.data["ShoppingMall"].fillna(self.data.groupby(["CryoSleep"])["ShoppingMall"].transform("median"))
        self.data["Spa"] = self.data["Spa"].fillna(self.data.groupby(["CryoSleep"])["Spa"].transform("median"))
        self.data["VRDeck"] = self.data["VRDeck"].fillna(self.data.groupby(["CryoSleep"])["VRDeck"].transform("median"))

        self.data["TotalBill"] = self.data["RoomService"] + self.data["FoodCourt"] + self.data["ShoppingMall"] + self.data["Spa"] + self.data["VRDeck"]

        bill_med = self.data["TotalBill"].mean()
        def fill_vip(row):
            if pd.isna(row["VIP"]):
                return int(row["TotalBill"] >= bill_med)
            return row["VIP"]
        
        self.data["VIP"] = self.data.apply(fill_vip, axis=1)

        dest = pd.get_dummies(self.data["Destination"], prefix="Hp").mul(self.data["TotalBill"], axis=0)
        self.data = self.data.join(dest)
    
    def __k_means(self):
        cluster_features = ["VRDeck", "Spa", "ShoppingMall", "FoodCourt", "RoomService", "TotalBill"]
        
        X_scaled = self.data.loc[:, cluster_features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        
        kmeans = KMeans(n_clusters=10, n_init=50, random_state=0)
        
        self.data["Cluster"] = kmeans.fit_predict(X_scaled)
        
        distances = kmeans.transform(X_scaled)
        
        distances_df = pd.DataFrame(
            distances, 
            columns=[f"Centroid{i}" for i in range(distances.shape[1])],
            index=self.data.index
        )
        
        self.data = self.data.join(distances_df)
    
    def data_processed(self):
        self.__id_processed()
        self.__location_processed()
        self.__name_processed()
        self.__cabin_processed()
        self.__age_processed()
        self.__service_processed()
        self.__k_means()
        self.data = self.data.drop(columns=["Name", "Active"])

    def save_csv(self) -> str:
        self.output = "processed_data.csv"
        output_file = os.path.join(self.datasets_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    datasets_path = "spaceship/datasets"
    data = Data(datasets_path)

    data.data_processed()
    
    data.mi_scores()
    data.baseline_score()
    data.save_csv()
