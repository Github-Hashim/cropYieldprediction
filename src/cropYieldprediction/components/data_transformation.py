import sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.cropYieldprediction.logger import logging
from src.cropYieldprediction.exception import CustomException
from src.cropYieldprediction.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_tranformer_obj(self):
        '''
        this function is responsilbe for data transformation
        '''
        try:
            numerical_columns = ["Rain Fall (mm)",'Fertilizer','Temperatue','Nitrogen (N)','Phosphorus (P)', 'Potassium (K)']
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            logging.info(f'Numerical columns: {numerical_columns}')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            CustomException(e,sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading the train and test file')
            preprocessing_obj = self.get_data_tranformer_obj()

            target_column_name = 'Yeild (Q/acre)'
            
            # Dividing the train dataset into dependent and independent feature
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Dividing the test dataset into dependent and independent feature
            input_feature_test_df = test_df.drop(target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing on training and test dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
