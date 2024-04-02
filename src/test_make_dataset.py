import sys
sys.path.append('..')

from src.utilities import Utilities

import pytest
import pandas as pd

@pytest.fixture
def params():
    """
    Read Parameters from params.yaml file
    """
    return Utilities().read_params()

def testing_if_dataframe_is_loaded():
    """
    testing_if_dataframe_is_loaded 

    This function is used to check if the dataset is present at specified url.
    """


    inital_url = params['data_location']['data_url_base']
    data_url = params['data_location']['data_url']

    url = inital_url + data_url.split('/')[-2]

    data = pd.read_csv(url)

    assert isinstance(data, pd.DataFrame)


