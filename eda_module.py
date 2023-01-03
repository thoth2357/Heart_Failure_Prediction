import logging
import pandas as pd



class Preprocessing:
    def __init__(self, dataset) -> None:
        self.data = dataset

        
    def check_missing_value(self):
        'purpose: checks for missing value in dataset'
        missing_value_count = self.data.isnull().sum().sum()  # chaining methods together
        if missing_value_count > 0:
            print(f'---> {missing_value_count} missing values found. Using interpolation method to fill missing values')
            self.interpolated_data = self.data.interpolate(method ='linear', 
            limit_direction ='forward')

            
            #More Rigid Process
            #check if misssing value is still present in interpolated data
            missing_value_count = self.interpolated_data.isnull().sum().sum()
            if missing_value_count > 0:
                print(f'---> {missing_value_count} missing values found after using initial interpolation method. Dropping rows of missing values')
                self.interpolated_data = self.interpolated_data.dropna()
            
            print(f'---> Missing Value problem solved..Data is clean and ready to use')
            return self.interpolated_data
        else:
            print(f'---> No Missing Value was found data is clean and ready to use')
            return self.data

    def descriptives(self, interpolated_data):
        'purpose:finds the descriptives of the datasets'
        print(f'---> Showing descriptive statistics of the dataset')
        print(pd.concat([interpolated_data.describe().T,
                      interpolated_data.median().rename('median'),
                      interpolated_data.skew().rename('skew'),
                      interpolated_data.kurt().rename('kurt'),
                     ], axis=1).T)


