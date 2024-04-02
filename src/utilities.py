import os
import yaml


class Utilities:
    def __init__(self,params_path = '../params.yaml') -> None:
        self.params_path = params_path
    
    def create_folder(self,*args):
        """
        create_folder 

        This function creates folder from the args given .
        The args must be given in specific order where the folder
        need to be created.

        """
        
        try:
            folder_path = '..'
            for arg in args:
                folder_path = os.path.join(folder_path,arg)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except Exception as e:
            print(f'error in creating folder:{e}')

    def read_params(self):
        """
        read_params 

        This function reads the parameters from yaml file

        Returns
        -------
        Dictionary
            Returns the dictionary with loaded values from file


        """
        try:
            print(self.params_path)
            with open(self.params_path,'r') as params_file:
                params = yaml.safe_load(params_file)
        except Exception as e:
            raise e
        else:
            return params

        
