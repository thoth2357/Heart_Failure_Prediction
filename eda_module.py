import logging
import pandas as pd
from IPython.display import display
import numpy as np
from scipy import stats


class Preprocessing:
    def __init__(self, dataset) -> None:
        self.data = dataset

    def check_missing_value(self):
        "purpose: checks for missing value in dataset"
        missing_value_count = (
            self.data.isnull().sum().sum()
        )  # chaining methods together
        if missing_value_count > 0:
            print(
                f"---> {missing_value_count} missing values found. Using interpolation method to fill missing values"
            )
            self.interpolated_data = self.data.interpolate(
                method="linear", limit_direction="forward"
            )

            # More Rigid Process
            # check if misssing value is still present in interpolated data
            missing_value_count = self.interpolated_data.isnull().sum().sum()
            if missing_value_count > 0:
                print(
                    f"---> {missing_value_count} missing values found after using initial interpolation method. Dropping rows of missing values"
                )
                self.interpolated_data = self.interpolated_data.dropna()

            print(f"---> Missing Value problem solved..Data is clean and ready to use")
            return self.interpolated_data
        else:
            print(f"---> No Missing Value was found data is clean and ready to use")
            return self.data

    def descriptives(self, interpolated_data):
        "purpose:finds the descriptives of the datasets"
        print(f"---> Showing descriptive statistics of the dataset")
        print(
            pd.concat(
                [
                    interpolated_data.describe().T,
                    interpolated_data.median().rename("median"),
                    interpolated_data.skew().rename("skew"),
                    interpolated_data.kurt().rename("kurt"),
                ],
                axis=1,
            ).T
        )


class FeatureSelection:
    """
    class for performing feature selection and significance testing
    """

    def __init__(self, dataset) -> None:
        self.data = dataset

    def chi_square_method(self, col1, col2):
        """
        chi_square method to check for association between two categorical variables
        """
        # ---create the contingency table---
        df_cont = pd.crosstab(index=self.data[col1], columns=self.data[col2]).values
        display(df_cont)
        # ---calculate degree of freedom---
        degree_f = (df_cont.shape[0] - 1) * (df_cont.shape[1] - 1)
        # ---sum up the totals for row and columns---
        df_cont.loc[:, "Total"] = df_cont.sum(axis=1)
        df_cont.loc["Total"] = df_cont.sum()
        print("---Observed (O)---")
        display(df_cont)
        # ---create the expected value dataframe---
        df_exp = df_cont.copy()
        df_exp.iloc[:, :] = (
            np.multiply.outer(df_cont.sum(1).values, df_cont.sum().values)
            / df_cont.sum().sum()
        )
        print("---Expected (E)---")
        display(df_exp)

        # calculate chi-square values
        df_chi2 = ((df_cont - df_exp) ** 2) / df_exp
        df_chi2.loc[:, "Total"] = df_chi2.sum(axis=1)
        df_chi2.loc["Total"] = df_chi2.sum()

        print("---Chi-Square---")
        display(df_chi2)
        # ---get chi-square score---
        chi_square_score = df_chi2.iloc[:-1, :-1].sum().sum()

        p = stats.distributions.chi2.sf(chi_square_score, degree_f)
        return chi_square_score, degree_f, p

    def mann_whitney(self, col1, col2):
        """
        mann_whitney method to check for association between two categorical variables
        """
        df1 = self.data[col1]
        df2 = self.data[col2]
        stat, p = stats.mannwhitneyu(df1, df2)
        return stat, p
