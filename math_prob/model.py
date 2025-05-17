# Updated on May 5, 2025, 10:10 PM
# Public Score: 0.7756
# Rank: 182/229

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.sparse import hstack, vstack

from preprocess import Data

class MathProblemModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        """
        Initialize the MathProblemModel instance

        Params:
            data_path: Relative path to the data directory
            data_processed_path: Relative path to the processed data CSV file
        """
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        
        data = pd.read_csv(data_path)

        self.train_data = data[data["id"] == -1].copy()
        self.test_data = data[data["id"] != -1].copy()
        self.text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        self.test_ids = self.test_data["id"].astype(int).values

        # Text features
        self.X_text_train = self.text_vectorizer.fit_transform(self.train_data["Question"])
        self.X_text_test = self.text_vectorizer.transform(self.test_data["Question"])

        # Numeric features
        numeric_cols = [col for col in data.columns if col not in ["Question", "label", "id"]]
        scaler = StandardScaler()
        self.X_num_train = scaler.fit_transform(self.train_data[numeric_cols])
        self.X_num_test = scaler.transform(self.test_data[numeric_cols])

        # Combine features
        self.X_train = hstack([self.X_text_train, self.X_num_train]).tocsr()
        self.y_train = self.train_data["label"].astype(int).values

        self.X_test = hstack([self.X_text_test, self.X_num_test]).tocsr()

    def predict(self) -> None:
        """
        Train a staged XGBoost classification pipeline and generate predictions
        """
        # Mapping from original labels to coarse topic groups
        self.topic_map = {
            0: 0,  # Algebra
            1: 1,  # Geometry/Trigonometry
            2: 2,  # Calculus/Analysis
            3: 2,  # Probability/Statistics
            4: 3,  # Number Theory
            5: 3,  # Combinatorics/Discrete
            6: 0,  # Linear Algebra
            7: 0   # Abstract Algebra/Topology
        }
        y_group = np.array([self.topic_map[y] for y in self.y_train])
        group_ids = sorted(set(y_group))

        # Stage-1 classifier: predict group assignments
        stage1_clf = XGBClassifier(eval_metric="mlogloss", verbosity=0, use_label_encoder=False)
        stage1_clf.fit(self.X_train, y_group)

        # Stage-2: train one XGB per group with label encoding
        models_stage2 = {}
        for group in group_ids:
            mask = (y_group == group)
            X_sub = self.X_train[mask]
            y_sub = self.y_train[mask]
            if len(y_sub) < 2:
                continue
            # Encode original labels to 0..(n_classes-1)
            unique_labels = np.unique(y_sub)
            lbl2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
            y_enc = np.array([lbl2idx[l] for l in y_sub])

            print(f"Training XGB for topic-group {group} with labels {list(unique_labels)}...")
            clf = XGBClassifier(eval_metric="mlogloss", verbosity=0, use_label_encoder=False)
            clf.fit(X_sub, y_enc)
            models_stage2[group] = (clf, unique_labels)

        # Predict on test set (stage-1)
        grp_pred = stage1_clf.predict(self.X_test)
        final_pred = np.zeros(grp_pred.shape, dtype=int)
        for group, (clf, unique_labels) in models_stage2.items():
            idx = np.where(grp_pred == group)[0]
            if idx.size == 0:
                continue
            X_sub = self.X_test[idx]
            pred_enc = clf.predict(X_sub)
            final_pred[idx] = [unique_labels[i] for i in pred_enc]

        # Pseudo-labeling: combine train + test using predicted labels
        y_pseudo = final_pred
        X_combined = vstack([self.X_train, self.X_test])
        # Concat original and pseudo labels
        y_combined = np.concatenate([self.y_train, y_pseudo])

        # Refit global XGB classifier on combined data
        final_model = XGBClassifier(eval_metric="mlogloss", verbosity=0, use_label_encoder=False)
        final_model.fit(X_combined, y_combined)
        # Final prediction
        global_predictions = final_model.predict(self.X_test)

        submission = pd.DataFrame({
            "id": self.test_ids,
            "label": global_predictions
        })
        
        # Save the submission CSV file.
        output_file = os.path.join(self.data_dir, "math_prob_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to math_prob_submission.csv!")


if __name__ == "__main__":
    data_path = "math_prob/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = MathProblemModel(data_path, data_processed_path)
    model.predict()

# Training XGB for topic-group 0 with labels [0, 6, 7]...
# Training XGB for topic-group 1 with labels [1]...
# Training XGB for topic-group 2 with labels [2, 3]...
# Training XGB for topic-group 3 with labels [4, 5]...
# Submission saved to math_prob_submission.csv!