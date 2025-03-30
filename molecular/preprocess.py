import os
import random
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

class Data:
    def __init__(self, datasets_path: str) -> None:
        """
        Initialize the Data instance by reading training and test CSV files

        Params:
            datasets_path: Relative path to the datasets directory
        """
        self.datasets_dir = os.path.join(os.getcwd(), datasets_path)

        # Build file paths for train and test datasets
        train_file_path = os.path.join(self.datasets_dir, "train.csv")
        test_file_path = os.path.join(self.datasets_dir, "test.csv")

        # Read CSV files into pandas DataFrames
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        # Concatenate training and test data for unified processing
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __drop_uninformative(self) -> None:
        """
        Drop columns that are deemed uninformative based on mutual information scores.
        This method retains only the columns with non-zero mutual information scores along with essential columns
        """
        # Retrieve columns with a positive mutual information score
        cols_to_keep = self.mi_scores_dataset().index.tolist()
        # Always keep these critical columns
        cols_to_keep += ["Batch_ID", "T80", "Smiles"]
        # Filter the dataframe to only include the selected columns
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        """
        Calculate mutual information scores for features in the training dataset

        Returns:
            pd.Series: A series containing mutual information scores sorted in descending order for features with scores greater than 0.
        """
        X = self.data[self.data["Batch_ID"].str.contains("Train")].copy()
        y = X.pop("T80")
        X.pop("Batch_ID")
        X.pop("Smiles")
            
        # Determine which features are discrete (integer) types
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        # Compute mutual information scores for each feature
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        mi_scores = mi_scores[mi_scores > 0]

        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        """
        Evaluate a baseline model using cross-validation on the training data.
        This method computes and prints the baseline Root Mean Squared Logarithmic Error (RMSLE)
        and Mean Absolute Error (MAE) on the log scale

        Params:
            model: An instance of the XGBRegressor
        """
        # Extract training data for baseline evaluation
        X = self.data[self.data["Batch_ID"].str.contains("Train")].copy()
        y = X.pop("T80")
        X.pop("Batch_ID")
        X.pop("Smiles")
        
        # Apply log transformation to the target variable to stabilize variance
        log_y = np.log(y)
        
        # Calculate RMSLE using cross-validation (negative MSE is converted back to positive and square-rooted)
        rmsle = np.sqrt(-cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_squared_error"
        ).mean())
        
        # Calculate Mean Absolute Error (MAE) on the log scale using cross-validation
        mae = -cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_absolute_error"
        ).mean()
        
        print(f"Baseline RMSLE: {rmsle:.5f}")
        print(f"Baseline MAE (log scale): {mae:.5f}")

    def __energy_processed(self) -> None:
        """
        Process energy-related features by computing statistical aggregates from T, S, and O columns
        """
        # Define column names for triplet states (T), singlet states (S), and oscilator strengths (O)
        T_cols = [f"T{i}" for i in range(1, 21)]
        S_cols = [f"S{i}" for i in range(1, 21)]
        O_cols = [f"O{i}" for i in range(1, 21)]
        
        # Compute basic statistics for T features
        self.data["T_mean"] = self.data[T_cols].mean(axis=1)
        self.data["T_std"]  = self.data[T_cols].std(axis=1)
        self.data["T_min"]  = self.data[T_cols].min(axis=1)
        self.data["T_max"]  = self.data[T_cols].max(axis=1)
        self.data["T_range"] = self.data["T_max"] - self.data["T_min"]
        
        # Compute basic statistics for S features
        self.data["S_mean"] = self.data[S_cols].mean(axis=1)
        self.data["S_std"]  = self.data[S_cols].std(axis=1)
        self.data["S_min"]  = self.data[S_cols].min(axis=1)
        self.data["S_max"]  = self.data[S_cols].max(axis=1)
        self.data["S_range"] = self.data["S_max"] - self.data["S_min"]
        
        # Compute the difference between the minimum values of S and T
        self.data["min_diff_S_T"] = self.data["S_min"] - self.data["T_min"]
        
        # Compute aggregate features for O features
        self.data["O_sum"] = self.data[O_cols].sum(axis=1)
        self.data["O_mean"] = self.data[O_cols].mean(axis=1)
        self.data["O_std"] = self.data[O_cols].std(axis=1)
    
    def __smiles_processed(self) -> None:
        """
        Process SMILES strings to generate RDKit descriptors and additional molecular properties
        """
        def gen_rdkit(df):
            # Retrieve SMILES strings from the dataframe
            smiles = df["Smiles"]
            # Prepare a numpy array to hold 10 RDKit-based descriptor values for each molecule
            RDKitProp = np.zeros(((smiles.shape[0]), 10), dtype=float)
            for d_key, smile in enumerate(smiles):
                # Convert SMILES to molecule and add hydrogens for 3D structure generation
                mol3d = Chem.AddHs(Chem.MolFromSmiles(smile))

                # Embed the molecule in 3D space and optimize geometry
                AllChem.EmbedMolecule(mol3d, randomSeed=random.randint(1, 1000000))
                AllChem.MMFFOptimizeMolecule(mol3d)

                # Calculate various molecular descriptors using RDKit
                RDKitProp[d_key, 0] = Descriptors.MolWt(mol3d)                        # Molecular weight
                RDKitProp[d_key, 1] = rdMolDescriptors.CalcNumHBA(mol3d)              # Number of hydrogen bond acceptors
                RDKitProp[d_key, 2] = rdMolDescriptors.CalcNumHBD(mol3d)              # Number of hydrogen bond donors
                RDKitProp[d_key, 3] = Descriptors.MolLogP(mol3d)                      # LogP (octanol-water partition coefficient)
                RDKitProp[d_key, 4] = rdMolDescriptors.CalcAsphericity(mol3d)         # Asphericity
                RDKitProp[d_key, 5] = rdMolDescriptors.CalcRadiusOfGyration(mol3d)    # Radius of gyration
                RDKitProp[d_key, 6] = Descriptors.TPSA(mol3d)                         # Topological Polar Surface Area
                RDKitProp[d_key, 7] = rdMolDescriptors.CalcNumRings(mol3d)            # Number of rings
                RDKitProp[d_key, 8] = rdMolDescriptors.CalcNumRotatableBonds(mol3d)   # Number of rotatable bonds
                RDKitProp[d_key, 9] = rdMolDescriptors.CalcNumHeteroatoms(mol3d)      # Number of heteroatoms
            
            # Append the computed descriptors as new columns to the dataframe
            df = pd.concat([df, pd.DataFrame(RDKitProp, columns=[f"RDKit_{i}" for i in range(10)])], axis=1)
            return df

        self.data = gen_rdkit(self.data)
        
        def compute_molecular_complexity(smiles):
            # Compute molecular complexity based on ring count, number of atoms, and logP
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan, np.nan, np.nan
            
            ring_count = Descriptors.RingCount(mol)
            num_atoms = mol.GetNumAtoms()
            logP = Descriptors.MolLogP(mol)
            
            # Weighted formula for molecular complexity
            complexity = ring_count * 0.5 + num_atoms * 0.3 + logP * 0.2
            
            return ring_count, num_atoms, complexity

        # Apply the molecular complexity computation to each SMILES string and create new columns
        self.data[["ring_count", "num_atoms", "complexity"]] = self.data["Smiles"].apply(
            lambda x: pd.Series(compute_molecular_complexity(x))
        )

        # Compute additional features using existing columns
        self.data["size_index"] = self.data["Mass"] / (self.data["Rg"] + 1e-6)  # Avoid division by zero
        self.data["hetero_polarity"] = self.data["NumHeteroatoms"] * self.data["TPSA"]
        self.data["Ncomplexity"] = self.data["NumHeteroatoms"] + self.data["Rg"] * 0.5 + self.data["TPSA"] * 0.1

    def __physical_processed(self) -> None:
        """
        Process physical properties by creating new features based on existing physical measurements
        """
        # Calculate ratio of hydrogen donors to acceptors, ensuring no division by zero
        self.data["ratio_HDonors_HAcceptors"] = self.data["HDonors"] / (self.data["HAcceptors"] + 1e-6)
        # Difference between mass and LogP to capture balance between mass and hydrophobicity
        self.data["mass_minus_logP"] = self.data["Mass"] - self.data["LogP"]
        # Total number of hydrogen bonds (donors + acceptors)
        self.data["total_H_bonds"] = self.data["HDonors"] + self.data["HAcceptors"]
    
    def __orbital_processed(self) -> None:
        """
        Process orbital properties to generate features related to energy levels and reactivity
        """
        # Calculate the energy gap between LUMO and HOMO levels
        self.data["energy_gap"] = self.data["LUMO(eV)"] - self.data["HOMO(eV)"]

        # Calculate differences in orbital energies from neighboring states
        self.data["delta_LUMO"] = self.data["LUMOp1(eV)"] - self.data["LUMO(eV)"]
        self.data["delta_HOMO"] = self.data["HOMO(eV)"] - self.data["HOMOm1(eV)"]

        # Compute ratios to capture relative changes in orbital energies
        self.data["ratio_LUMO"] = self.data["delta_LUMO"] / (self.data["energy_gap"] + 1e-6)
        self.data["ratio_HOMO"] = self.data["delta_HOMO"] / (self.data["energy_gap"] + 1e-6)

        # Calculate hardness and softness as measures of chemical reactivity
        self.data["hardness"] = self.data["energy_gap"] / 2.0
        self.data["softness"] = 1 / (self.data["hardness"] + 1e-6)

        # Compute chemical potential as the average of HOMO and LUMO energies
        self.data["chemical_potential"] = (self.data["HOMO(eV)"] + self.data["LUMO(eV)"]) / 2.0

        # Electrophilicity index derived from chemical potential and hardness
        self.data["electrophilicity"] = (
            (self.data["chemical_potential"] * self.data["chemical_potential"])
            / (2.0 * self.data["hardness"])
        )

        # Combined orbital feature that might capture overall orbital interaction effects
        self.data["combined_orbital_feature"] = self.data["delta_LUMO"] - self.data["delta_HOMO"]

    def __tricky_predict(self) -> None:
        """
        Compute a prediction value using a complex formula based on various features

        Note: The idea for this formula is from Spiritmilk
        """
        self.data["predict"] = (
            (self.data["T8"] - self.data["T2"] * self.data["TDOS3.2"] * self.data["LUMOp1(eV)"]) * (1.7020332667085079 * self.data["SDOS3.9"])
            +
            (
                ((self.data["TDOS3.7"] - self.data["SDOS3.8"] - self.data["TDOS4.3"] - self.data["TDOS2.7"])
                / ((self.data["T14"] / self.data["TDOS3.6"]) + (0.8998248512734 - self.data["RDKit_8"])))
                +
                ((self.data["LUMO(eV)"] + self.data["T2"] * self.data["TDOS3.2"] * self.data["LUMOp1(eV)"]) * ((self.data["LUMO(eV)"] + self.data["T2"] * self.data["TDOS3.2"] * self.data["LUMOp1(eV)"]) - self.data["TDOS3.2"]))
            )
        )
        
    def data_processed(self) -> None:
        """
        Process the data by executing energy, SMILES, physical, orbital feature engineering,
        computing a tricky prediction, and dropping uninformative features
        """
        self.__energy_processed()
        self.__smiles_processed()
        self.__physical_processed()
        self.__orbital_processed()
        self.__tricky_predict()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        """
        Save the processed data to a CSV file in the datasets directory

        Returns:
            str: The filename of the saved CSV
        """
        self.output = "processed_data.csv"
        output_file = os.path.join(self.datasets_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    datasets_path = "molecular/datasets"
    data = Data(datasets_path)
    
    data.data_processed()
    data.baseline_score_dataset()
    data.save_csv()

#[17:15:13] UFFTYPER: Unrecognized atom type: Se2+2 (24)
#[17:15:19] UFFTYPER: Unrecognized atom type: Se2+2 (31)
#[17:15:23] UFFTYPER: Unrecognized atom type: Se2+2 (23)
#Baseline RMSLE: 1.26260
#Baseline MAE (log scale): 1.09878
#Data saved to processed_data.csv!
