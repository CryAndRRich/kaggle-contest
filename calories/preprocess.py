import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor
pd.set_option("future.no_silent_downcasting", True)

class Data():
    def __init__(self, data_path: str) -> None:
        """
        Initialize Data object by loading raw train and test CSVs, concatenating them,
        and performing initial missing-value handling

        Parameters:
            data_path: Path to the directory containing "train1.csv", "train2.csv" and "test.csv"
        """
        self.data_dir = os.path.join(os.getcwd(), data_path)

        # Load training and testing data
        train_file_path1 = os.path.join(self.data_dir, "train1.csv")
        train_file_path2 = os.path.join(self.data_dir, "train2.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data1 = pd.read_csv(train_file_path1)
        train_data2 = pd.read_csv(train_file_path2)
        test_data = pd.read_csv(test_file_path)

        # Combine data for unified preprocessing
        self.data = pd.concat([train_data1, test_data], axis=0, ignore_index=True)

        train_data2 = train_data2.rename(columns={"User_ID": "id", "Gender": "Sex"})
        self.data = pd.concat([train_data2, self.data], axis=0, ignore_index=True)

        for col in self.data.select_dtypes(include=["int64", "float64"]).columns:
            self.data[col] = self.data[col].fillna(0).astype("float64")

        self.data["Sex"] = self.data["Sex"].astype("category")
        self.data["Calories"] = np.log1p(self.data["Calories"])

    def __drop_uninformative(self) -> None:
        mi_scores = self.mi_scores_dataset()
        cols_to_keep = mi_scores[mi_scores > 0.0].index.tolist()
        cols_to_keep += ["id", "Calories"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]
        print(cols_to_keep)

    def mi_scores_dataset(self) -> pd.Series:
        df = self.data[(self.data["id"] < 750000) | (self.data["id"] > 5000000)].copy()
        y = df.pop("Calories") 
        df.pop("id") 
        
        # Factorize categorical columns
        for colname in df.select_dtypes(["object", "category"]):
            df[colname], _ = df[colname].factorize()

        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df.dtypes]
        mi = mutual_info_regression(df, y, discrete_features=discrete_features, random_state=0)

        mi_scores = pd.Series(mi, index=df.columns, name="MI Scores").sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        df = self.data[(self.data["id"] < 750000) | (self.data["id"] > 5000000)].copy()
        y = df.pop("Calories")
        df.pop("id")

        # Convert categorical features to integer codes
        for colname in df.select_dtypes(["category"]):
            df[colname] = df[colname].cat.codes

        scoring = {
            "MAE":   "neg_mean_absolute_error",
            "RMSE":  "neg_mean_squared_error"
        }

        for name, scorer in scoring.items():
            score = cross_val_score(model, df, y, cv=5, scoring=scorer).mean()
            if name == "RMSE":
                score = np.sqrt(-score)
            else: 
                score = -score
            print(f"Baseline {name}: {score:.5f}")
            
    def __body_processed(self) -> None:
        self.data["BMI"] = self.data["Weight"] / np.square(self.data["Height"] / 100.0)
        self.data["BMI_Category"] = pd.cut(
            self.data["BMI"], bins=[0, 18.5, 24.9, 29.9, 100], 
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        ).astype("category")
        
        # Du Bois calculation
        self.data["BSA"] = 0.007184 * np.power(self.data["Weight"], 0.425) * np.power(self.data["Height"], 0.725)
        
        # Mifflin-St Jeor calculation
        bmr = 10 * self.data["Weight"] + 6.25 * self.data["Height"] - 5 * self.data["Age"]
        self.data["BMR"] = np.where(self.data["Sex"] == 0, bmr - 161, bmr + 5)
        
        # Nes calculation
        self.data["Max_Heart_Rate"] = 211 - 0.64 * self.data["Age"]
        self.data["Intensity"] = self.data["Heart_Rate"] / self.data["Max_Heart_Rate"]
        self.data["HR_Zone"] = pd.cut(
            self.data["Intensity"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
            labels=["Very Light", "Light", "Moderate", "Hard", "Very Hard", "Max"
        ]).astype("category")
        self.data["HR_Duration"] = self.data["Heart_Rate"] * self.data["Duration"]
        self.data["HR_Weight"] = self.data["Heart_Rate"] * self.data["Weight"]
        self.data["HR_Duration_Weight"] = self.data["Heart_Rate"] * self.data["Duration"] * self.data["Weight"]
        self.data["Body_Strain_Index"] = (self.data["Heart_Rate"] * self.data["Body_Temp"]) / self.data["Weight"]
        
        def calc_tdee(row):
            intensity = row["Intensity"]
            bmr = row["BMR"]
            if intensity >= 0.5 and intensity < 0.6:
                return bmr * 1.2
            elif intensity >= 0.6 and intensity < 0.7:
                return bmr * 1.375
            elif intensity >= 0.7 and intensity < 0.8:
                return bmr * 1.55
            elif intensity >= 0.8 and intensity < 0.9:
                return bmr * 1.725
            else:
                return bmr * 1.9
        self.data["TDEE"] = self.data.apply(calc_tdee, axis = 1)

        # Metabolic Efficiency Index
        self.data["Metabolic_Efficiency"] = self.data["BMR"] * (self.data["Heart_Rate"] / self.data["BMR"].median())
        
        # Thermic Effect Ratio
        self.data["Thermic_Effect"] = (self.data["Body_Temp"] * 100) / (self.data["Weight"] ** 0.5)
        
        # Power Output Estimate
        self.data["Power_Output"] = self.data["Weight"] * self.data["Duration"] * (self.data["Heart_Rate"] / 1000)
        self.data["Work_Done"] = self.data["Power_Output"] * self.data["Duration"]

    def data_processed(self) -> None:
        self.__body_processed()
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
    data_path = "calories/data"
    data = Data(data_path)

    data.data_processed()
    data.save_csv()
    data.baseline_score_dataset()

# Baseline MAE: 0.03650
# Baseline RMSE: 0.06168