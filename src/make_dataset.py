import os
import pandas as pd

import sys
sys.path.append('..')

from src.utilities import Utilities
from src.logfile import get_logger



logger = get_logger('make_dataset')


class MakeDataset:
    def __init__(self) -> None:
        pass

    def load_and_save(self ,filename, url):
        """
        load_and_save 

        This functions loads the data from specified url and stores the data in raw folder.

        Parameters
        ----------
        filename : string
            The name of the file to be stored
        url : string
            url link where the data is present
        """

        try:

            data = pd.read_csv(url)
            logger.info("new file is being created")
            


            main_data_folder = params['Data_paths']['main_data_path']
            raw_data_folder = params['Data_paths']['raw_data_path']


            Utilities().create_folder(main_data_folder)
            Utilities().create_folder(main_data_folder,raw_data_folder)

            with open(os.path.join('../', main_data_folder, raw_data_folder, str(filename)+'.csv'), 'w', newline='', encoding='utf-8') as file:
                data.to_csv(file, index=False, sep=',')

        except Exception as e:
            logger.error(e)
            raise(e)
        

        
    def save_data(self,data ,file_name,data_type):
        """
        save_data 

        This function is used to save the data from pandas dataframe as csv file
        in particular location based on given type train,test,processed,cleaned

        Parameters
        ----------
        data : Dataframe
            Pandas Dataframe which is to be stored
        file_name : string
            Name on which file is to be stored
        data_type : string
            ['cleaned','train','test','processed','other']
        """
        try:
            params = Utilities().read_params()

            main_data_folder = params['Data_paths']['main_data_path']
            raw_data_folder = params['Data_paths']['raw_data_path']

            Utilities().create_folder(main_data_folder)
            if data_type == 'cleaned':
                raw_data_folder = params['Data_paths']['cleaned_data_path']
            elif data_type == 'train':
                raw_data_folder = params['Data_paths']['train_data_path']
            elif data_type == 'test':
                raw_data_folder = params['Data_paths']['test_data_path']
            elif data_type == 'processed':
                raw_data_folder = params['Data_paths']['processed_data_path']
            else:
                raw_data_folder = params['Data_paths']['other']

            Utilities().create_folder(main_data_folder,raw_data_folder)

            data.to_csv(os.path.join('..',main_data_folder, raw_data_folder, str(
                        file_name+'.csv')), index=False, sep=',')
        except Exception as e:
            logger.error(e)
            raise(e)
        
    def load_data(self ,folder_name,file_name):
        """
        load_data 

        This function is used to load data from csv file to pandas
        dataframe in the given folder 

        Parameters
        ----------
        folder_name : string
            The name of folder in Data where the file is present
        file_name : string
            Name of the file that is to be loaded into dataframe

        Returns
        -------
        returns Dataframe loaded from csv file
            
        """

        try:
            params = Utilities().read_params()

            main_data_folder = params['Data_paths']['main_data_path']
            

            file_path = os.path.join('../',main_data_folder,folder_name,file_name+'.csv')

            data = pd.read_csv(file_path)
            return data


        except Exception as e:
            print(f'error in load_Data:{e}')
            logger.error(e)
            raise(e)



        
if __name__ == "__main__":


    params = Utilities().read_params()

    data_url = params['data_location']['data_url']

    md = MakeDataset()
    logger.info('Loading of data started.')

    raw_data_filename = params['Data_paths']['raw_data_filename']
    md.load_and_save(raw_data_filename,data_url)
    logger.info('Data loading completed and data saved to the directory Data/raw')






        