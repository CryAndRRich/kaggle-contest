import os
import re
import numpy as np
import pandas as pd

class Data():
    def __init__(self, data_path: str) -> None:
        """
        Initialize the Data object by loading raw train and test CSVs,
        concatenating them, fitting a TF-IDF on all questions, and
        performing initial missing-value handling

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

        # Initial fill for numerical columns
        for col in ["id", "label"]:
            self.data[col] = self.data[col].fillna(-1)

        # Ensure Question is string
        self.data["Question"] = self.data["Question"].astype(str)

    def check_stats(self):
        """
        Display label distribution statistics for the training portion

        This method filters out rows where "label" == -1 (test data),
        computes counts and percentage distribution of each label,
        and prints the results
        """
        df = self.data[self.data["label"] != -1]
        counts = df["label"].value_counts().sort_index()
        total = counts.sum()
        percent = (counts / total * 100).round(2)
        stats = pd.DataFrame({"count": counts, "percent": percent})
        print("Label distribution (count and %):")
        print(stats)

    def __text_processed(self) -> None:
        """
        Clean and normalize math-related text in the "Question" column

        This private method applies a series of regex-based transformations
        to replace math symbols with descriptive tokens, remove LaTeX-like
        commands, and normalize whitespace and case
        """

        def clean_math_text(text):
            text = str(text)
            
            # Mapping of math symbol patterns to descriptive replacements
            math_symbols = {
                r"\$": " dollar_expr ",
                r"\=": " equals ",
                r"\<": " less than ",
                r"\>": " greater than ",
                r"\+": " plus ",
                r"\-": " minus ",
                r"\*": " times ",
                r"\/": " divided by ",
                r"\^": " to the power of ",
                r"\√": " square root ",
                r"\π": " pi_expr ",
                r"\∑": " sum_expr ",
                r"\∫": " integral_expr ",
                r"\∞": " infinity_expr "
            }
            
            for pattern, replacement in math_symbols.items():
                text = re.sub(pattern, replacement, text)
            
            # Remove other LaTeX commands and braces
            text = re.sub(r"\\[a-zA-Z]+", " ", text)
            text = re.sub(r"\{([^}]*)\}", r" \1 ", text)
            
            # Remove non-alphanumeric characters (except punctuation)
            text = re.sub(r"[^a-zA-Z0-9\s\.\?\!]", " ", text)
            
            # Separate alphanumeric tokens
            text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", text)  
            text = re.sub(r"([a-zA-Z])(\d+)", r"\1 \2", text)  
            
            # Normalize whitespace and lowercase
            text = re.sub(r"\s+", " ", text).strip().lower()
            
            return text
            
        self.data["Question"] = self.data["Question"].apply(clean_math_text)

    def __math_processed(self):
        """
        Extract numeric and domain-specific features from the cleaned text

        This private method computes basic length features, counts of
        numeric tokens, and high-level counts for various math topics
        """
        self.data["length"] = self.data["Question"].astype(str).apply(len)
        self.data["log_length"] = np.log1p(self.data["length"])

        # Count of math-related symbols
        self.data["numeric_token_counts"] = self.data["Question"].str.findall(r"[\d+\-*/=()<>^√π÷%\.]").str.len()
        self.data["log_numeric_token_counts"] = np.log1p(self.data["numeric_token_counts"])
        
        def extract_math_features(text):
            # Domain-specific feature counts
            features = {
                "num_count": len(re.findall(r"\d+", text)),
                "equation_count": len(re.findall(r"equals|equation|formula|solve for", text)),
                "function_count": len(re.findall(r"function|f\(x\)|derivative|integral", text)),
                "geometry_count": len(re.findall(r"angle|triangle|circle|area|volume", text)),
                "algebra_count": len(re.findall(r"variable|polynomial|matrix|vector", text)),
                "calculus_count": len(re.findall(r"derivative|integral|limit|differentiation", text)),
                "prob_count": len(re.findall(r"probability|random|random variable", text)),
                "topo_count": len(re.findall(r"ring|abelian|nonabelian", text)),
                "word_count": len(text.split()),
                "is_proof": int("prove" in text or "show that" in text),
                "is_compute": int("compute" in text or "calculate" in text),
                "is_find": int("find" in text or "determine" in text),
                "is_matrix": int("matrix" in text or "matrices" in text),
            }
            return pd.Series(features)

        math_features = self.data["Question"].apply(extract_math_features)
        self.data = pd.concat([self.data, math_features], axis=1)

    def data_processed(self) -> None:
        self.__text_processed()
        self.__math_processed()

    def save_csv(self) -> str:
        """
        Save the processed data to a CSV file in the data directory

        Returns:
            str: The filename of the saved CSV
        """
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"

if __name__ == "__main__":
    data_path = "math_prob/data"
    data = Data(data_path)

    data.data_processed()
    data.save_csv()
    data.check_stats()

# Data saved to processed_data.csv!
# Label distribution (count and %):
#        count  percent
# label
# 0.0     2618    25.69
# 1.0     2439    23.94
# 2.0     1039    10.20
# 3.0      368     3.61
# 4.0     1712    16.80
# 5.0     1827    17.93
# 6.0      100     0.98
# 7.0       86     0.84