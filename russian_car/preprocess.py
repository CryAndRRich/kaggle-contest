import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from datasets.supplemental_english import *

class Data:
    def __init__(self, datasets_path: str) -> None:
        """
        Initialize the Data object by loading train and test datasets and concatenating them

        Parameters:
            datasets_path: The relative path to the datasets directory containing "train.csv" and "test.csv"
        """
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)

        # Build file paths for train and test CSV files
        train_file_path = os.path.join(self.datasets_dir, "train.csv")
        test_file_path = os.path.join(self.datasets_dir, "test.csv")

        # Read CSV files into DataFrames
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        # Concatenate train and test datasets into a single DataFrame
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __drop_uninformative(self) -> None:
        """
        Drop uninformative columns from the dataset
        """
        # Get the list of informative columns based on mutual information scores
        cols_to_keep = self.mi_scores_dataset().index.tolist()
        cols_to_keep += ["id", "price"]  # Ensure "id" and "price" columns are preserved
        # Filter the data to include only the selected columns
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        """
        Calculate and return the mutual information scores for features in the dataset

        Returns:
            pd.Series: A sorted Series of mutual information scores for each feature
        """
        
        # Select only the training portion based on the "id" threshold
        X = self.data[self.data["id"] <= 51635].copy()
        y = X.pop("price")  # Remove "price" column to use as the target variable
        X.pop("id")         # Remove "id" column as it"s not a predictive feature

        # Factorize categorical columns to convert them into numerical values
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
            
        # Create a list indicating which features are discrete (integer type)
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        # Compute mutual information scores between features and target "price"
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        mi_scores = mi_scores[mi_scores > 10e-6]  # Filter out near-zero scores

        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        """
        Evaluate the baseline performance of a regression model on the dataset
        Parameters:
            model: The regression model to use for evaluation
        """

        # Use only the training data based on the "id" threshold
        X = self.data[self.data["id"] <= 51635].copy()
        y = X.pop("price")  # Extract target variable
        X.pop("id")         # Remove non-predictive "id" column
        
        # Apply logarithmic transformation to stabilize variance
        log_y = np.log(y)

        # Factorize categorical features
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        
        # Compute Root Mean Squared Log Error (RMSLE) using cross-validation
        rmsle = np.sqrt(-cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_squared_error"
        ).mean())
        
        # Compute Mean Absolute Error (MAE)
        mae = -cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error"
        ).mean()

        # Define SMAPE function to compute symmetric mean absolute percentage error
        def smape(y_true, y_pred):
            y_true_orig = np.exp(y_true)
            y_pred_orig = np.exp(y_pred)
            return np.mean(2 * np.abs(y_pred_orig - y_true_orig) / (np.abs(y_true_orig) + np.abs(y_pred_orig) + 1e-8)) * 100

        smape_scorer = make_scorer(smape, greater_is_better=False)

        # Compute SMAPE score using cross-validation
        smape_score = -cross_val_score(model, X, y, cv=5, scoring=smape_scorer).mean()

        # Print the baseline metrics for evaluation
        print(f"Baseline RMSLE: {rmsle:.5f}")
        print(f"Baseline MAE: {mae:.5f}")
        print(f"Baseline SMAPE: {smape_score:.5f}")
    
    def __plate_processed(self):
        """
        Process the "plate" column to extract and generate new features

        This method splits the plate string into multiple components:
        first_letter, number, second_letter, region_code, and a combined string (first_letter + second_letter).
        It also generates binary features to indicate if there are repeated or mirrored letters and numbers.
        The original "plate" column is dropped after processing
        """
        def split_plate(row):
            # Split the plate into its components based on character positions
            first_letter = row[0]
            number = row[1:4]
            second_letter = row[4:6]
            region_code = row[6:]
            # Combine first and second letters for additional analysis
            return first_letter, int(number), second_letter, int(region_code), first_letter + second_letter

        # Apply the split_plate function to every entry in the "plate" column and create new columns
        self.data[["first_letter", "number", "second_letter", "region_code", "combine"]] = self.data["plate"].apply(
            lambda x: pd.Series(split_plate(x))
        )

        # Remove the original "plate" column after processing
        self.data = self.data.drop(columns=["plate"])

        def has_repeated(x):
            x = str(x)
            # Return 1 if there are repeated characters, 0 otherwise
            return int(len(set(x)) < len(x))
        
        def has_sequential_numbers(num):
            n_str = str(num).zfill(3)  # Ensure the number is 3 digits with leading zeros if necessary
            seq_up = ["123", "234", "345", "456", "567", "678", "789"]
            seq_down = ["987", "876", "765", "654", "543", "432", "321"]
            # Check for any sequential pattern in either ascending or descending order
            return any(seq in n_str for seq in seq_up) or any(seq in n_str for seq in seq_down)
        
        def has_mirror(x):
            x = str(x)
            # Return 1 if the string is the same forwards and backwards, indicating a mirror effect
            return int(x == x[::-1])
        
        # Create binary features for letter patterns and number patterns
        self.data["has_repeated_letters"] = self.data["combine"].apply(has_repeated)
        self.data["has_mirror_letters"] = self.data["combine"].apply(has_mirror)

        self.data["has_repeated_numbers_number"] = self.data["number"].apply(has_repeated)
        self.data["has_sequential_numbers_number"] = self.data["number"].apply(has_sequential_numbers)
        self.data["has_mirror_numbers_number"] = self.data["number"].apply(has_mirror)

        self.data["has_repeated_numbers_region_code"] = self.data["region_code"].apply(has_repeated)
        self.data["has_sequential_numbers_region_code"] = self.data["region_code"].apply(has_sequential_numbers)
        self.data["has_mirror_numbers_region_code"] = self.data["region_code"].apply(has_mirror)

    def __date_processed(self):
        """
        Process the "date" column and generate new time-related features

        The method converts the "date" column to datetime format, extracts year, month, and day,
        computes the number of days, months, and years from the initial listing per vehicle (grouped by "combine"),
        creates a flag for listings at the end of the year, and assigns a ranking for each listing
        """
        import pandas as pd

        # Convert "date" column to datetime object
        self.data["date"] = pd.to_datetime(self.data["date"])

        # Extract basic date components
        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month
        self.data["day"] = self.data["date"].dt.day

        # Calculate the number of days since the first listing for each vehicle (grouped by "combine")
        self.data["days_from_initial_listing"] = self.data.groupby("combine")["date"].transform("min")
        self.data["days_from_initial_listing"] = (self.data["date"] - self.data["days_from_initial_listing"]).dt.days
        
        # Convert days to months and years (approximate)
        self.data["months_from_initial_listing"] = round(self.data["days_from_initial_listing"] / 30, 3)
        self.data["years_from_initial_listing"] = round(self.data["months_from_initial_listing"] / 12, 3)

        # Flag listings that occur in December (end of the year)
        self.data["year_end"] = (self.data["month"] == 12).astype(int)

        # Assign a ranking number to each listing per vehicle based on the listing date
        self.data["listing_num"] = self.data.groupby("combine")["date"].rank(method="dense").astype(int)
        # Convert the datetime back to date for easier readability
        self.data["date"] = self.data["date"].dt.date
    
    def __supplemental_processed(self):
        """
        Process supplemental data to add additional features based on region and government codes

        The method applies functions to determine the region name and government-related details
        (advantage on road and significance) by matching the region code and plate combination against
        predefined mappings (i.e., REGION_CODES and GOVERNMENT_CODES). If no match is found, default values are assigned.
        """
        def supple_region_detail(row):
            region_code = row["region_code"]

            # Iterate through REGION_CODES mapping to find the matching region name
            for region_name, codes in REGION_CODES.items():
                if str(region_code) in codes:
                    return region_name
            return "None"
            
        def supple_government_detail(row):
            letter = row["combine"]
            region_code = row["region_code"]

            # Iterate through GOVERNMENT_CODES mapping to get government details
            for (code_letters, _, region), details in GOVERNMENT_CODES.items():
                if letter == code_letters and region_code == int(region):
                    return details[2], details[3]
            
            return 0, 0

        # Map the region detail to a new column "region_name"
        self.data[["region_name"]] = self.data.apply(
            lambda x: pd.Series(supple_region_detail(x)),
            axis=1
        )

        # Map government details to new columns "advantage_on_road" and "significance"
        self.data[["advantage_on_road", "significance"]] = self.data.apply(
            lambda x: pd.Series(supple_government_detail(x)),
            axis=1
        )

    def __price_processed(self):
        """
        Process the "price" column and engineer additional price-related features

        This method applies a logarithmic transformation to the price,
        flags prestigious vehicles based on certain patterns in the plate features,
        and computes the average price for various groupings such as region, number, first letter, second letter, 
        and combination. These computed averages are mapped back onto the dataset
        """
        # Apply log1p transformation to smooth out the price distribution
        self.data["price"] = np.log1p(self.data["price"])

        available = "ABEKMHOPCTYX"
        exceptional_str = set()
        # Generate exceptional patterns where the first and third characters are the same
        for i in available:
            for j in available:
                exceptional_str.add(i + j + i)
        exceptional_num = set([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 
            10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 200, 300, 400, 500, 600, 700, 800, 900, 
            123, 234, 345, 456, 567, 678, 789, 
            987, 876, 765, 654, 543, 432, 321
        ])
        # Add additional exceptional number patterns based on a simple formula
        for i in range(1, 10):
            for j in range(10):
                exceptional_num.add(i * 100 + j * 10 + i)

        def check_higher_price(row):
            combine = row["combine"]
            number = row["number"]
            region_code = row["region_code"]
            # Check if the plate patterns match any of the exceptional criteria
            if combine in exceptional_str or number in exceptional_num or region_code in exceptional_num:
                return 1
            return 0
        
        # Create a binary "prestigious" flag based on plate patterns
        self.data[["prestigious"]] = self.data.apply(
            lambda x: pd.Series(check_higher_price(x)),
            axis=1
        )

        # Compute average prices for various groupings; drop rows with missing price
        valid_prices = self.data.dropna(subset=["price"])

        # Average price by region code
        avg_price_by_region = valid_prices.groupby("region_code")["price"].mean()
        self.data["avg_price_by_region"] = self.data["region_code"].map(avg_price_by_region)

        # Average price by number component of the plate
        avg_price_by_number = valid_prices.groupby("number")["price"].mean()
        self.data["avg_price_by_number"] = self.data["number"].map(avg_price_by_number)

        # Average price by first letter of the plate
        avg_price_by_first_letter = valid_prices.groupby("first_letter")["price"].mean()
        self.data["avg_price_by_first_letter"] = self.data["first_letter"].map(avg_price_by_first_letter)
        
        # Average price by second letter of the plate
        avg_price_by_second_letter = valid_prices.groupby("second_letter")["price"].mean()
        self.data["avg_price_by_second_letter"] = self.data["second_letter"].map(avg_price_by_second_letter)

        # Combine first and second letter averages to create a composite feature
        self.data["avg_price_by_letter"] = np.sqrt(
            self.data["avg_price_by_first_letter"] * self.data["avg_price_by_second_letter"]
        )

        # Average price by the combined letter pattern
        avg_price_by_combine = valid_prices.groupby("combine")["price"].mean()
        self.data["avg_price_by_combine"] = self.data["combine"].map(avg_price_by_combine)

        # Average price by year of listing
        avg_price_by_year = valid_prices.groupby("year")["price"].mean()
        self.data["avg_price_by_year"] = self.data["year"].map(avg_price_by_year)

        # Average price based on the prestigious flag
        avg_price_by_prestigious = valid_prices.groupby("prestigious")["price"].mean()
        self.data["avg_price_by_prestigious"] = self.data["prestigious"].map(avg_price_by_prestigious)

    def data_processed(self) -> None:
        """
        Run all processing methods to engineer features and drop uninformative columns
        """
        self.__plate_processed()
        self.__date_processed()
        self.__supplemental_processed()
        self.__price_processed()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        """
        Save the processed dataset to a CSV file

        Returns:
            str: The name of the output CSV file
        """
        self.output = "processed_data.csv"
        output_file = os.path.join(self.datasets_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    datasets_path = "russian_car/datasets"
    data = Data(datasets_path)
    
    data.data_processed()
    data.baseline_score_dataset()
    data.save_csv()

#Baseline RMSLE: 0.04979
#Baseline MAE: 0.43341
#Baseline SMAPE: 40.21632
#Data saved to processed_data.csv!
