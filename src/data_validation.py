import os
import sys
sys.path.append('..')
import pandas as pd

from src.utilities import Utilities
from src.logfile import get_logger

from pydantic import BaseModel , conint 
from typing import List , Optional,Dict





logger = get_logger('data_validation')



class Dictvalidator(BaseModel):
    """
    Dictvalidator

    this functions validates the type of data present in particular column

    Parameters
    ----------
    BaseModel : 
        Inherits the properties from Pydantic.BaseModel
    """


    tenure : int
    MonthlyCharges : float
    TotalCharges : float
    gender : str
    SeniorCitizen : str
    Partner : str
    Dependents : str
    PhoneService : str
    PaperlessBilling :str
    MultipleLines : str
    InternetService :str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaymentMethod:str



class Dataframevalidator(BaseModel):
    """
    Dataframevalidator 

    Checks for all the dataframe

    """
    df_dict : List[Dictvalidator]


if __name__ == '__main__':

    params = Utilities().read_params()
    main_data_folder = params['Data_paths']['main_data_path']
    raw_data_folder = params['Data_paths']['raw_data_path']
    raw_data_filename = params['Data_paths']['raw_data_filename']

    raw_data_file_path = os.path.join('..',main_data_folder,raw_data_folder,raw_data_filename+'.csv')
    data = pd.read_csv(raw_data_file_path)

    try : 
        logger.info('getting ready to open file')
        Dataframevalidator(df_dict = data.to_dict(orient='records'))
        logger.info('test completed')
    except Exception as e :
        logger.warning(e)
        raise(e)

