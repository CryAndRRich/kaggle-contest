import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by fixing some column values and renaming columns.

    Parameters:
        df: The raw input data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Replace "Brk Cmn" with "BrkComm" in the Exterior2nd column.
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    
    # For GarageYrBlt, replace corrupt values (> 2010) with the YearBuilt value.
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    
    # Rename columns that start with a number to a more manageable format.
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
        "FireplaceQu": "FireplaceQual"
    }, inplace=True)
    
    return df

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.
    For numeric columns, use the median; for categorical columns, use "None".

    Parameters:
        df: The DataFrame to impute.

    Returns:
        pd.DataFrame: The DataFrame with imputed missing values.
    """
    # Impute numeric columns with the median value.
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(df[name].median())
    
    # Impute categorical columns with "None".
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    
    return df

# List of nominal (unordered) categorical features.
features_nom = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", 
    "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", 
    "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", 
    "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", 
    "MiscFeature", "SaleType", "SaleCondition"
]

# Levels for ordinal encoding.
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQual": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Add a "None" level for missing values for all ordinal features.
ordered_levels = {key: ["None"] + value for key, value in ordered_levels.items()}

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features in the DataFrame.
    For nominal features, set them as category dtype and add a "None" category.
    For ordinal features, set them as ordered categorical types based on defined levels.

    Parameters:
        df: The DataFrame to encode.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical features.
    """
    # Process nominal categories.
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a "None" category for missing values if not already present.
        if "None" not in df[name].cat.categories:
            df[name] = df[name].cat.add_categories("None")
    
    # Process ordinal categories using the specified levels.
    for name, levels in ordered_levels.items():
        try:
            df[name] = df[name].astype(pd.CategoricalDtype(levels, ordered=True))
        except Exception as e:
            # If conversion fails, skip the feature.
            continue
    return df

class Data:
    def __init__(self, datasets_path: str) -> None:
        """
        Initialize the Data object by loading datasets, cleaning, encoding, and imputing missing values.

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
        
        # Apply cleaning, encoding, and imputation.
        self.data = clean(self.data)
        self.data = encode(self.data)
        self.data = impute(self.data)

    def __drop_uninformative(self) -> None:
        """
        Remove features that are deemed uninformative based on mutual information scores.
        This method retains columns with MI scores greater than a small threshold and the 'Id' and 'SalePrice' columns.
        """
        mi_scores = self.mi_scores_dataset()
        # Keep features with MI score above 10e-6.
        cols_to_keep = mi_scores[mi_scores > 10e-6].index.tolist()
        cols_to_keep += ['Id', 'SalePrice']
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        """
        Compute mutual information scores for features with respect to SalePrice for the training dataset.

        Returns:
            pd.Series: Mutual information scores indexed by feature names.
        """
        # Filter training data (Id <= 1460).
        X = self.data[self.data["Id"] <= 1460].copy()
        y = X.pop("SalePrice")
        X.pop("Id")
        
        # Factorize categorical columns.
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        
        # Determine which features are discrete.
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        # Calculate mutual information scores.
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        """
        Evaluate and print the baseline RMSLE and MAE (log scale) using cross-validation
        for the training dataset using the specified model.

        Parameters:
            model: The regression model to evaluate. Defaults to XGBRegressor().
        """
        # Filter training data.
        X = self.data[self.data["Id"] <= 1460].copy()
        y = X.pop("SalePrice")
        X.pop("Id")
        
        # Convert categorical features to integer codes.
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
        
        # Use logarithm of SalePrice.
        log_y = np.log(y)
        
        # Calculate Root Mean Squared Log Error (RMSLE).
        rmsle = np.sqrt(-cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_squared_error"
        ).mean())
        
        # Calculate Mean Absolute Error (MAE) on the log scale.
        mae = -cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_absolute_error"
        ).mean()
        
        print(f"Baseline RMSLE: {rmsle:.5f}")
        print(f"Baseline MAE (log scale): {mae:.5f}")

    def __age_processed(self) -> None:
        """
        Process and create new age-related features:
        - HouseAge: The age of the house at the time of sale.
        - HouseRemodAge: The number of years since the house was remodeled.
        - HouseCond: A condition score based on HouseAge.
        - GarageAge: The age of the garage.
        """
        self.data["HouseAge"] = np.maximum(self.data["YrSold"] - self.data["YearBuilt"], 0)
        self.data["HouseRemodAge"] = np.maximum(self.data["YrSold"] - self.data["YearRemodAdd"], 0)

        # Create a house condition score from the HouseAge.
        self.data["HouseCond"] = (self.data["HouseAge"] / 20).round().astype(int).clip(upper=4)
        self.data["HouseCond"] = self.data["HouseCond"].astype(pd.CategoricalDtype(list(range(5)), ordered=True))

        self.data["GarageAge"] = self.data["YrSold"] - self.data["GarageYrBlt"]

    def __area_processed(self) -> None:
        """
        Process and create new area-related features:
        - TotalArea: Sum of above-ground living area and basement area.
        - TotalSF: Total square footage calculated from different floor areas.
        - Ratios: Various ratios (living area to total area, living area to lot area, etc.).
        - Log-transformed features for GrLivArea and MasVnrArea.
        """
        self.data["TotalArea"] = self.data["GrLivArea"] + self.data["TotalBsmtSF"]
        self.data["TotalSF"] = self.data["FirstFlrSF"] + self.data["SecondFlrSF"] + self.data['BsmtFinSF1'] + self.data['BsmtFinSF2']
        
        self.data["LivAreaRatio"] = self.data["GrLivArea"] / self.data["TotalSF"]
        self.data["LivLotRatio"] = self.data["GrLivArea"] / self.data["LotArea"]
        self.data["RmsPerSqft"] = self.data["TotRmsAbvGrd"] / self.data["GrLivArea"]
        self.data["BedroomRatio"] = self.data["BedroomAbvGr"] / self.data["GrLivArea"]
        self.data["FlrRatio"] = self.data["FirstFlrSF"] / (self.data["FirstFlrSF"] + self.data["SecondFlrSF"])
        self.data["Spaciousness"] = (self.data["FirstFlrSF"] + self.data["SecondFlrSF"]) / self.data["TotRmsAbvGrd"]

        # Create one-hot encoded building type features multiplied by living area.
        bldg_area = pd.get_dummies(self.data["BldgType"], prefix="Bldg").mul(self.data["GrLivArea"], axis=0)
        self.data = self.data.join(bldg_area)

        # Calculate median living area per neighborhood.
        self.data["MedNhbdArea"] = self.data.groupby("Neighborhood", observed=False)["GrLivArea"].transform("median")

        # Log-transform living area and masonry veneer area.
        self.data["LgGrLivArea"] = np.log1p(self.data["GrLivArea"])
        self.data["LgMasVnrArea"] = np.log1p(self.data["MasVnrArea"])

    def __quality_processed(self) -> None:
        """
        Process quality-related features:
        - Convert OverallQual and OverallCond to numeric values.
        - Fill missing OverallQual using a prediction model.
        - Create composite quality scores and related features.
        - Process various quality features using a defined quality mapping.
        """
        # Remove "None" category and convert to numeric.
        self.data["OverallQual"] = self.data["OverallQual"].cat.remove_categories("None")
        self.data["OverallQual"] = pd.to_numeric(self.data["OverallQual"], errors="coerce")
        self.data["OverallCond"] = pd.to_numeric(self.data["OverallCond"], errors="coerce")

        # Fill missing OverallQual values.
        self._fill_overallqual()
        
        # Create overall average quality and other composite features.
        self.data["OverallAvg"] = ((self.data["OverallQual"] + self.data["OverallCond"]) / 2).round().astype("int")
        self.data["QualSF"] = self.data["TotalSF"] * self.data["OverallQual"]
        self.data["NhbQual"] = self.data.groupby("Neighborhood", observed=False)["OverallQual"].transform("median")

        # Convert quality features to categorical.
        self.data["OverallAvg"] = self.data["OverallAvg"].astype("category")
        self.data["OverallQual"] = self.data["OverallQual"].astype("category")
        self.data["OverallCond"] = self.data["OverallCond"].astype("category")

        quality_features = ["ExterQual", "KitchenQual", "BsmtQual", "FireplaceQual", "GarageQual"]
        qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

        # Map quality features to numeric scores and fill missing values.
        for feat in quality_features:
            self.data[feat] = self.data[feat].astype(str).str.strip().replace("NA", np.nan)
            self.data[feat + "Num"] = self.data[feat].map(qual_mapping)
            median_val = self.data[feat + "Num"].median()
            self.data[feat + "Num"] = self.data[feat + "Num"].fillna(median_val)
        
        # Calculate composite quality as the mean of numeric quality scores.
        num_cols = [feat + "Num" for feat in quality_features]
        self.data["CompositeQual"] = self.data[num_cols].mean(axis=1)

        # Re-convert quality features back to categorical.
        self.data["ExterQual"] = self.data["ExterQual"].astype("category")
        self.data["KitchenQual"] = self.data["KitchenQual"].astype("category")
        self.data["BsmtQual"] = self.data["BsmtQual"].astype("category")
        self.data["FireplaceQual"] = self.data["FireplaceQual"].astype("category")
        self.data["GarageQual"] = self.data["GarageQual"].astype("category")
    
    def _fill_overallqual(self) -> None:
        """
        Predict and fill missing OverallQual values using a RandomForestRegressor model.
        This method uses several related features to predict the missing overall quality.
        """
        # Features used for predicting OverallQual.
        features = [
            "ExterQual", "BsmtQual", "KitchenQual", "Neighborhood", 
            "GrLivArea", "GarageArea", "YearBuilt", "YearRemodAdd"
        ]

        # Separate rows with known and missing OverallQual values.
        known = self.data[self.data["OverallQual"].notna()].copy()
        unknown = self.data[self.data["OverallQual"].isna()].copy()
        
        # One-hot encode features for the known data.
        X_train = pd.get_dummies(known[features])
        y_train = known["OverallQual"]
        
        # Train a RandomForestRegressor model.
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # One-hot encode features for the unknown data.
        X_unknown = pd.get_dummies(unknown[features])
        X_unknown = X_unknown.reindex(columns=X_train.columns, fill_value=0)
        
        # Predict missing OverallQual values.
        predicted = model.predict(X_unknown)
        
        # Fill in the missing OverallQual values (rounded to nearest integer).
        self.data.loc[self.data["OverallQual"].isna(), "OverallQual"] = np.round(predicted).astype(int)
    
    def __basement_processed(self) -> None:
        """
        Process basement-related features:
        - Map basement quality and condition to numeric values.
        - Calculate a basement quality score.
        """
        qual_mapping = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.data['BsmtQual'] = self.data['BsmtQual'].astype(str)
        self.data['BsmtCond'] = self.data['BsmtCond'].astype(str)
        self.data['BsmtQualNum'] = self.data['BsmtQual'].map(qual_mapping)
        self.data['BsmtCondNum'] = self.data['BsmtCond'].map(qual_mapping)
        
        # Convert basement quality columns to categorical.
        self.data["BsmtQual"] = self.data["BsmtQual"].astype("category")
        self.data["BsmtCond"] = self.data["BsmtCond"].astype("category")
        
        # Fill missing numeric mappings with 0 and compute a combined quality score.
        self.data['BsmtQualNum'] = self.data['BsmtQualNum'].fillna(0)
        self.data['BsmtCondNum'] = self.data['BsmtCondNum'].fillna(0)
        self.data['BsmtQualityScore'] = np.round((self.data['BsmtQualNum'] + self.data['BsmtCondNum']) / 2.0).astype(int)
    
    def __porch_processed(self) -> None:
        """
        Process porch-related features by creating a feature that counts the number of porch types present.
        """
        self.data["PorchTypes"] = self.data[[ 
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "Threeseasonporch",
            "ScreenPorch",
        ]].gt(0.0).sum(axis=1)

    def __bath_processed(self) -> None:
        """
        Create a feature for the total number of bathrooms by combining full and half baths (above and below ground).
        """
        self.data["TotalBath"] = (
            self.data["FullBath"] +
            0.5 * self.data["HalfBath"] +
            self.data["BsmtFullBath"] +
            0.5 * self.data["BsmtHalfBath"]
        )

    def __k_means(self) -> None:
        """
        Perform KMeans clustering on selected features to create new cluster-based features.
        This method adds a 'Cluster' label for each row and also appends distances to each cluster centroid.
        """
        # Define features to use for clustering.
        cluster_features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea"]
        
        # Standardize the features.
        X_scaled = self.data.loc[:, cluster_features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        
        # Initialize KMeans with 20 clusters and fixed random state.
        kmeans = KMeans(n_clusters=20, n_init=50, random_state=0)
        
        # Fit KMeans and assign cluster labels.
        self.data["Cluster"] = kmeans.fit_predict(X_scaled)
        
        # Calculate distances to each centroid.
        distances = kmeans.transform(X_scaled)
        
        # Create a DataFrame for centroid distances.
        distances_df = pd.DataFrame(
            distances, 
            columns=[f"Centroid{i}" for i in range(distances.shape[1])],
            index=self.data.index
        )
        
        # Join the centroid distance features back into the main dataset.
        self.data = self.data.join(distances_df)

    def data_processed(self) -> None:
        """
        Run all processing methods to engineer features and drop uninformative columns.
        """
        self.__age_processed()
        self.__area_processed()
        self.__quality_processed()
        self.__basement_processed()
        self.__porch_processed()
        self.__bath_processed()
        self.__k_means()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        """
        Save the processed DataFrame to a CSV file in the datasets directory.

        Returns:
            str: The filename of the saved CSV.
        """
        self.output = "processed_data.csv"
        output_file = os.path.join(self.datasets_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    datasets_path = "housing_price/datasets"
    data = Data(datasets_path)
    
    data.data_processed()
    data.baseline_score_dataset()
    data.save_csv()

# Baseline RMSLE: 0.13483
# Baseline MAE (log scale): 0.09351
# Data saved to processed_data.csv!
