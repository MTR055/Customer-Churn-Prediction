import sys
sys.path.append('..')

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.utilities import Utilities
from src.logfile import get_logger



params = Utilities().read_params()

logger = get_logger('web')

class WebApp:

    def __init__(self) -> None:
        pass

    def webapp(self):
        """
        This method is used to create a webapp by which users will be able to make predictions

        """

        try:
            st.set_page_config(
                page_title="Churn Detection",
                page_icon="🔄",
                layout="wide",
                initial_sidebar_state="expanded",
            )
            # Adding the title to the page
            st.title('Customer Churn Prediction')

            # Adding a author name to the project
            st.caption('By M.Thirupati Reddy')

            # Making Predictions
            st.header('Make Prediction')

            # Creating an interfact to get inputs from the user
            col1, col2, col3 = st.columns(3)

             

            TotalCharges = col1.number_input('TotalCharges', min_value=0.00,
                                max_value=10000.00)
            MonthlyCharges = col1.number_input('MonthlyCharges', min_value=0.00,
                                max_value=2000.00)
            tenure = col1.number_input('tenure(days)', min_value=0,
                                max_value=2000)

            gender = col1.selectbox(
                'gender', ['Male', 'Female'])
            SeniorCitizen = col1.selectbox(
                'SeniorCitizen', ['Yes', 'No'])
            Partner = col1.selectbox(
                'Partner', ['Yes', 'No'])
            Dependents = col2.selectbox('Dependents', ['Yes', 'No'])
    
            PhoneService = col2.selectbox('PhoneService', ['Yes', 'No'])
            PaperlessBilling = col2.selectbox('PaperlessBilling', ['Yes', 'No'])

            MultipleLines = col2.selectbox('MultipleLines', ['No', 'Yes','No phone service'])
            OnlineSecurity = col2.selectbox('OnlineSecurity', [ 'Yes','No','No internet service'])
            OnlineBackup = col2.selectbox('OnlineBackup', [ 'Yes','No' , 'No internet service'])
            DeviceProtection = col2.selectbox('DeviceProtection', [ 'Yes','No','No internet service'])
            TechSupport = col3.selectbox('TechSupport', [ 'Yes','No','No internet service'])
            StreamingTV = col3.selectbox('StreamingTV', [ 'Yes','No','No internet service'])
            StreamingMovies = col3.selectbox('StreamingMovies', ['Yes','No','No internet service'])
            

            InternetService = col3.selectbox('InternetService', ['Fiber optic', 'DSL','No'])
            Contract = col3.selectbox('Contract', ['One year','Two year','Month-to-month'])
            PaymentMethod = col3.selectbox('PaymentMethod', ['Electronic check', 'Mailed check','Bank transfer (automatic)','Credit card (automatic)'])

            input = np.array([[gender, SeniorCitizen, Partner, Dependents,tenure, PhoneService, MultipleLines, 
                               InternetService,OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,StreamingTV,
                                 StreamingMovies, Contract, PaperlessBilling,PaymentMethod, MonthlyCharges, TotalCharges]])

            input1 = pd.DataFrame(input, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                                                    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                                                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                                    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

            predict = st.button('Make a Prediction')

            # Actions after user clicks on 'Make a Prediction' button
            if predict:
                with st.spinner('Please wait'):
                    preprocess_pipe_foldername = params['preprocess']['preprocessing_main_folder']
                    preprocess_pipe_filename = params['preprocess']['preprocess_pipe_filename']

                    model_foldername = params['Model_paths']['model_path']
                    model_name = params['Model_paths']['model_name']

                    # Loading saved preprocess pipeline
                    with open(os.path.join('..',preprocess_pipe_foldername, preprocess_pipe_filename), 'rb') as f:
                        preprocess_pipeline = joblib.load(f)

                    # Loading the saved machine learning model
                    def load_model(model_foldername, model_name):
                        model = joblib.load(os.path.join('..',
                            model_foldername, model_name))
                        return model

                    model = load_model(model_foldername, model_name)

                    # Preprocessing the input provided by the user
                    transformed_input = preprocess_pipeline.transform(input1)

                    # Making predictions using the saved model and the preprocessed data
                    prediction = model.predict(transformed_input)
                    print(prediction)

                    # making the predictions understandable for the user
                    churn_dict = dict()
                    churn_dict[0] = 'Customer continues'
                    churn_dict[1] = 'Customer Left'
                    

                    prediction = churn_dict[prediction[0]]

                    # Showing the prediction made to the user
                    st.subheader(f"Customer status:   {prediction}")

        except Exception as e:
            logger.error(e)
            raise e


if __name__ == "__main__":
    wa = WebApp()
    wa.webapp()