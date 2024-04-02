import os
import pandas as pd

import sys
sys.path.append('..')

from src.utilities import Utilities

from src.logfile import get_logger
from src.utilities import Utilities
from src.make_dataset import MakeDataset
from src.data_preprocessing import preprocessing

from xgboost import XGBClassifier
from sklearn.metrics import  recall_score,classification_report,balanced_accuracy_score,f1_score,precision_score
from sklearn.pipeline import Pipeline
import joblib
import json
import mlflow

logger = get_logger('model_training')

class Model_Training:
    def __init__(self) -> None:
        pass

    def train_model(self):
        """
        train_model 

        This function reads the training data and testing data from the files and 
        parameters of the model from params folder and trains the model on training data 
        and stores classification report into metric file and stores the trained model
        as joblib file in Models folder.
        """

        try:

            params = Utilities().read_params()
            md = MakeDataset()
            pp = preprocessing()


            x_train = md.load_data('train_data','x_train')
            y_train = md.load_data('train_data','y_train')
            x_test = md.load_data('test_data','x_test')
            y_test = md.load_data('test_data','y_test')

            x_train_processed ,y_train_processed = pp.transforming_data(x_train,y_train)
            x_test_processed ,y_test_processed = pp.transforming_data(x_test,y_test)

            mlflow.autolog()
            with mlflow.start_run():
                max_depth = params['model']['params']['max_depth']
                n_estimators = params['model']['params']['n_estimators']
                max_leaves = params['model']['params']['max_leaves']
                learning_rate = params['model']['params']['learning_rate']

                mlflow.log_param('max_depth', max_depth)
                mlflow.log_param('n_estimators', n_estimators)
                mlflow.log_param('max_leaves', max_leaves)
                mlflow.log_param('learning_rate', learning_rate)


                xgboost_pipe = XGBClassifier(
                    max_depth=max_depth, n_estimators=n_estimators, max_leaves=max_leaves, learning_rate=learning_rate)
                logger.info('Model initialized')



                # Fitting the model on train data
                xgbc = xgboost_pipe.fit(x_train_processed, y_train_processed)
                logger.info('Model trained on the train data.')


                # Predicting metrics using the trained model and the test data
                y_pred = xgbc.predict(x_test_processed)


                balanced_accuracy_scr = balanced_accuracy_score(y_test_processed, y_pred)
                p_scr = precision_score(y_test_processed, y_pred, average='weighted')
                r_scr = recall_score(y_test_processed, y_pred, average='weighted')
                f1_scr = f1_score(y_test_processed, y_pred, average='weighted')
                clf_report = classification_report(
                    y_test_processed, y_pred, output_dict=True)
                clf_report = pd.DataFrame(clf_report).transpose()

                mlflow.log_metric('balanced_accuracy_score',
                                    balanced_accuracy_scr)
                mlflow.log_metric('precision_score', p_scr)
                mlflow.log_metric('recall_score', r_scr)
                mlflow.log_metric('f1_score', f1_scr)

                logger.info(
                    'Trained model evaluation done using validation data.')

            # Saving the calculated metrics into a json file in the Metrics folder
            metrics_folder = params['Model_paths']['metric_main_path']
            metrics_filename = params['Model_paths']['metrics_filename']


            Utilities().create_folder(metrics_folder)

            with open(os.path.join('..',metrics_folder, metrics_filename), 'w') as json_file:
                metrics = dict()
                metrics['balanced_accuracy_score'] = balanced_accuracy_scr
                metrics['precision_score'] = p_scr
                metrics['recall_score'] = r_scr
                metrics['f1_score'] = f1_scr

                json.dump(metrics, json_file, indent=4)

            clf_report_path = params['Model_paths']['clf_report_filename']

            clf_report.to_csv(os.path.join('..',metrics_folder, clf_report_path))

            logger.info('Saved evaluations in files.')

            # Saving the trained machine learing model in the models folder
            model_foldername = params['Model_paths']['model_path']
            model_name = params['Model_paths']['model_name']


            Utilities().create_folder(model_foldername)

            model_dir = os.path.join('..',model_foldername, model_name)

            joblib.dump(xgbc, model_dir)

            logger.info('Trained model saved as a joblib file.')
        except Exception as e:
            print(f'error in model_training:{e}')
            logger.error(e)
            raise(e)
        

if __name__ == '__main__':
    Model_Training().train_model()

    