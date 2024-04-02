import os
import logging
import datetime

import sys
sys.path.append('..')

from src.utilities import Utilities



def get_logger(filename):
    """
    get_logger 

    This function initiates the logging and store the information of logging files generated 
    from respective file on particular date which helps in finding the actions performed on 
    particular date.

    Parameters
    ----------
    filename : string
        The name of the file in which the class is called without .py. Ensure the name of the file
        is present in params file to generate logging file. If not add particular filename in params file.


    Returns
    -------
    .logs
    
        Stores the logs generated from particular file on particular date in Logs folder.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    Utilities().create_folder('Logs')
    params = Utilities().read_params()

    main_log_folderpath = params['logging_folder_paths']['main_log_foldername']
    make_data_path = params['logging_folder_paths'][filename]
    Utilities().create_folder(main_log_folderpath,make_data_path)

    current_date = datetime.date.today()

    file_handler = logging.FileHandler(os.path.join('../',main_log_folderpath, make_data_path,str(current_date)))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

        

    
