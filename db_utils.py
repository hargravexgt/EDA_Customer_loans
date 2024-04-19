import yaml
import pandas as pd
from sqlalchemy import create_engine

def credentials_to_dict():
    with open('credentials.yaml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data

class RDSDatabaseConnector:
    def __init__(self, dict_credentials) -> None:
        self.dict_credentials = dict(dict_credentials)

    def init_sql_engine(self):
        db_user = self.dict_credentials['RDS_USER']
        db_password = self.dict_credentials['RDS_PASSWORD']
        db_host = self.dict_credentials['RDS_HOST']
        db_port = self.dict_credentials['RDS_PORT']
        db_name = self.dict_credentials['RDS_DATABASE']
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(db_url)
        

    def extract_loan_payments_data(self):
        sql_query = "SELECT * FROM loan_payments"
        df = pd.read_sql_query(sql_query, self.engine)
        return df

def save_as_csv(df, file_name):
    df.to_csv(f'{file_name}.csv', index = False)

def read_csv_to_df(file_path):
    df = pd.read_csv(file_path)
    return df

