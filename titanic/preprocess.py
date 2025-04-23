import os
from typing import Tuple, Any

import numpy as np
import pandas as pd

class Data:
    """
    A class for processing the Titanic dataset by merging the training and test sets
    and applying various feature engineering steps.
    """
    def __init__(self, datasets_path: str) -> None:
        """
        Initialize the Data object with the path to the datasets. It loads both
        train and test CSV files and concatenates them.

        Parameters:
            datasets_path: The relative path to the directory containing the datasets.
        """
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)

        train_file_path = os.path.join(self.datasets_dir, "train.csv")
        test_file_path = os.path.join(self.datasets_dir, "test.csv")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __name_processed(self) -> None:
        """
        Process the "Name" column to extract titles and adjust them based on the passenger"s age.
        It also creates the "IsAdult" column.
        """
        # Extract titles from names
        self.data["Title"] = self.data["Name"].str.extract(
            r",\s*(Mrs|Mr|Master|Miss|Don|Rev|Dr|Ms|Major|Lady|Sir|Col|Mme|Mlle|Capt|the Countess|Jonkheer)",
            expand=False
        )

        # Map certain titles to common ones
        male_mask = self.data["Title"].isin(["Don", "Rev", "Dr", "Major", "Sir", "Col", "Capt", "Jonkheer"])
        female_mask = self.data["Title"].isin(["Lady", "Ms", "Mme", "Mlle", "the Countess"])
        self.data.loc[male_mask, "Title"] = "Mr"
        self.data.loc[female_mask, "Title"] = "Mrs"

        # Create the IsAdult column using the _check_adult method
        self.data["IsAdult"] = self.data.apply(self._check_adult, axis=1)

        # Adjust titles based on adult status
        self.data.loc[(self.data["Title"] == "Miss") & (self.data["IsAdult"] == 1), "Title"] = "Mrs"
        self.data.loc[(self.data["Title"] == "Mrs") & (self.data["IsAdult"] == 0), "Title"] = "Miss"
        self.data.loc[(self.data["Title"] == "Master") & (self.data["IsAdult"] == 1), "Title"] = "Mr"
        self.data.loc[(self.data["Title"] == "Mr") & (self.data["IsAdult"] == 0), "Title"] = "Master"

    def _check_adult(self, row: pd.Series) -> int:
        """
        Determine if a passenger is an adult based on the Age and Title.

        Parameters:
            row: A row of the DataFrame.

        Returns:
            int: 1 if the passenger is considered an adult, 0 otherwise.
        """
        if pd.notnull(row["Age"]):
            return 1 if row["Age"] >= 21 else 0
        else:
            return 0 if row["Title"] in ["Master", "Miss"] else 1

    def __age_processed(self) -> None:
        """
        Process the "Age" column by filling in missing values based on the average age of
        each title group. It also calculates adjusted average ages.
        """
        self.avg_age: dict = {}
        for title in ["Mr", "Mrs", "Master", "Miss"]:
            valid_ages = self.data[(self.data["Title"] == title) & (self.data["Age"].notnull())]["Age"]
            mean_age = valid_ages.mean()

            def get_avg(age: float, male: bool) -> float:
                # Round the age value in a custom way
                age = int(age * 10) // 10
                if age - int(age) < 0.5:
                    return int(age) + 0.5 if male else int(age)
                else:
                    return int(age + 1) if male else int(age) + 0.5

            if title in ["Mrs", "Miss"]:
                self.avg_age[title] = get_avg(mean_age, False)
            else:
                self.avg_age[title] = get_avg(mean_age, True)

        self.data["Age"] = self.data.apply(self._fill_age, axis=1)

    def _fill_age(self, row: pd.Series) -> int:
        """
        Fill missing Age values based on the title"s average age or a default value.

        Parameters:
            row: A row from the DataFrame.

        Returns:
            int: The original Age if present, or the imputed Age.
        """
        if pd.notnull(row["Age"]):
            return row["Age"]
        title = row["Title"]
        if title in self.avg_age and not np.isnan(self.avg_age[title]):
            return self.avg_age[title]
        return 28

    def __sex_processed(self) -> None:
        """
        Process the "Sex" column by converting to lowercase and mapping to numerical values.
        """
        self.data["Sex"] = self.data["Sex"].str.lower().map({"male": 1, "female": 0})

    def __relative_processed(self) -> None:
        """
        Create family size features:
        - "FamSize" is the sum of SibSp, Parch, and 1 (the passenger itself).
        - "hasNanny" is set to 1 if the passenger is a child traveling alone.
        """
        self.data["FamSize"] = self.data["SibSp"] + self.data["Parch"] + 1

        self.data["hasNanny"] = 0
        self.data.loc[(self.data["Title"].isin(["Master", "Miss"])) & (self.data["FamSize"] == 1), "hasNanny"] = 1

    def __ticket_processed(self) -> None:
        """
        Process the "Ticket" column by splitting it into "TicketLetter" and "TicketNumber".
        """
        self.data["TicketLetter"], self.data["TicketNumber"] = zip(*self.data["Ticket"].apply(self._split_ticket))

    def _split_ticket(self, ticket: Any) -> Tuple[str, str]:
        """
        Split the ticket string into a letter part and a number part.

        Parameters:
            ticket: The ticket information as a string.

        Returns:
            Tuple[str, str]: A tuple containing the ticket letter and ticket number.
        """
        if not isinstance(ticket, str) or not ticket.strip():
            return "No", "0"
        items = ticket.split()
        ticket_item = items[0] if len(items) > 1 else "No"
        ticket_number = items[-1]
        return ticket_item, ticket_number

    def __fare_processed(self) -> None:
        """
        Process the "Fare" column by replacing zeros and missing values with the average fare of the passenger"s class.
        """
        self.data.loc[self.data["Fare"] == 0, "Fare"] = 1
        self.avg_fare_by_pclass = self.data.groupby("Pclass")["Fare"].mean()
        self.data["Fare"] = self.data.apply(self._fill_fare, axis=1)

    def _fill_fare(self, row: pd.Series) -> int:
        """
        Fill missing Fare values using the average fare of the corresponding passenger class.

        Parameters:
            row: A row from the DataFrame.

        Returns:
            int: The original Fare if available, or the imputed Fare.
        """
        if pd.isnull(row["Fare"]):
            return self.avg_fare_by_pclass[row["Pclass"]]
        else:
            return row["Fare"]

    def __cabin_processed(self) -> None:
        """
        Process the "Cabin" column by filling missing values and splitting it into "CabinLet" and "CabinNum".
        """
        self.data["Cabin"] = self.data["Cabin"].fillna("No")
        self.data[["CabinLet", "CabinNum"]] = self.data["Cabin"].apply(self._fill_cabin)

    def _fill_cabin(self, cabin: str) -> pd.Series:
        """
        Split the cabin string into a cabin letter and a cabin number.

        Parameters:
            cabin: The cabin information.

        Returns:
            pd.Series: A Series with the cabin letter and cabin number.
        """
        parts = cabin.split()
        if len(parts) < 2 and parts[-1].lower() == "no":
            return pd.Series(["N", -1])
        elif len(parts) < 2:
            let = parts[-1][0]
            num_str = parts[-1][1:]
            num_val = 0 if num_str == "" else int(num_str)
            return pd.Series([let, num_val])
        else:
            temp_letters = []
            temp_numbers = []
            for part in parts:
                if part == "F":
                    temp_letters.append("F")
                    temp_numbers.append(0)
                else:
                    temp_letters.append(part[0])
                    num_str = part[1:]
                    temp_numbers.append(int(num_str) if num_str else 0)
            if "F" in temp_letters:
                return pd.Series(["F", 0])
            else:
                avg_num = int(sum(temp_numbers) / len(temp_numbers))
                return pd.Series([temp_letters[-1], avg_num])

    def __embarked_processed(self) -> None:
        """
        Process the "Embarked" column by filling missing values with the mode.
        """
        self.data["Embarked"] = self.data["Embarked"].fillna(self.data["Embarked"].mode()[0])

    def data_processed(self) -> None:
        """
        Execute all processing steps to clean and transform the data. After processing,
        drop columns that are no longer needed.
        """
        self.__name_processed()
        self.__age_processed()
        self.__sex_processed()
        self.__relative_processed()
        self.__ticket_processed()
        self.__fare_processed()
        self.__cabin_processed()
        self.__embarked_processed()
        self.data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

    def save_csv(self) -> str:
        """
        Save the processed data to a CSV file named "processed_data.csv" in the datasets directory.

        Returns:
            str: The name of the output CSV file.
        """
        self.output = "processed_data.csv"
        output_file = os.path.join(self.datasets_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    datasets_path = "titanic/datasets"
    data = Data(datasets_path)
    data.data_processed()
    data.save_csv()
