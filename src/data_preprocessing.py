import os
import sys
sys.path.append('..')

import joblib

from src.utilities import Utilities
from src.logfile import get_logger
from src.make_dataset import MakeDataset

from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import OrdinalEncoder , StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



logger = get_logger('data_preprocessing')

class preprocessing:
    def __init__(self) -> None:
        pass

    def storing_train_and_test_dataset(self):
        """
        storing_train_and_test_dataset 
         
        This function is used to load raw data stored in the folder and apply some transformation and 
        split the data into training and testing datasets and store them in respective folders in Data .
        """
       

        try:

            md = MakeDataset()
            params = Utilities().read_params()
            main_data_folder = params['Data_paths']['main_data_path']
            raw_data_folder = params['Data_paths']['raw_data_path']
            raw_data_file = params['Data_paths']['raw_data_filename']
            target_col = [params['Data_paths']['target_column']]
            Id_col = ['customerID']

            
            data = md.load_data(raw_data_folder,raw_data_file)


            data = data[data['TotalCharges']!=' ']
            data.loc[:, 'TotalCharges'] = data.loc[:, 'TotalCharges'].astype(float)
            data.loc[:,'SeniorCitizen'] = ['Yes' if i == 1 else 'No' for i in data.loc[:,'SeniorCitizen']]
            
            X = data.drop([target_col[0],Id_col[0]], axis=1)
            y = data[target_col[0]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            md.save_data(X_train,'x_train','train')
            md.save_data(y_train,'y_train','train')
            md.save_data(X_test,'x_test','test')
            md.save_data(y_test,'y_test','test')

        except Exception as e:
            logger.error(e)
            raise(e)



    def transforming_data(self,X_train,y_train):
        """
        transforming_data 

        
        The function transforms the categorical and numeriacal
        columns in the given x-train and y-train arguments for using them to train model.


        Parameters
        ----------
        X_train : Pandas Dataframe
            Training Dataset
        y_train : Pandas Dataframe
            Training Datast of target variable

        Returns
        -------
        arrayd
            The columns in given dataset undergo transformations required to able to fit into
            the model and returns the transformed arrays.
        """
        
        try:

            params = Utilities().read_params()


            target_col = [params['Data_paths']['target_column']]


            cat_cols = X_train.nunique()[X_train.nunique() < 6].keys().tolist()
            cat_cols = [x for x in cat_cols if x not in target_col]
            num_cols = [x for x in X_train.columns if x not in cat_cols + target_col ]


            bin_cols = X_train.nunique()[X_train.nunique() == 2].keys().tolist()


            bin_cols = [i for i in bin_cols if i not in target_col]
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            


            column_transforming = ColumnTransformer([
                                        ('binary_cols',OrdinalEncoder(),bin_cols),
                                        ('multi_cols',OneHotEncoder(),multi_cols),
                                        ('numeric_cols',StandardScaler(),num_cols)
                                    ],remainder='passthrough')
            
            
            column_transforming.fit(X_train)
        
            X_train_processed  =  column_transforming.transform(X_train)
            y_train_processed = [1 if i == 'Yes' else 0 for i in y_train['Churn']]

            print(column_transforming)

            joblib.dump(column_transforming, '../preprocessing_pipelines/column_transformer.pkl')

            return X_train_processed , y_train_processed

            
        except Exception as e:
            logger.error(e)
            print(f'error in data transforming_data function:{e}')
            raise(e)
            
            
if __name__  == '__main__':
    try:
        preprocessing().storing_train_and_test_dataset()
        md = MakeDataset()
        x_train = md.load_data('train_data','x_train')
        y_train = md.load_data('train_data','y_train')

        x_train_preprocessed , y_train_preprocessed = preprocessing().transforming_data(x_train,y_train)

        loaded_column_transformer = joblib.load('../preprocessing_pipelines/column_transformer.pkl')

        print(loaded_column_transformer)
        logger.info('successfully transformed training dataset x and y')
    except Exception as e:
        print(f'error in processing notebook:{e}')

    



      





