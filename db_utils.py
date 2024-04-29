#%%

import yaml
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import yeojohnson
import plotly.express as px

def credentials_to_dict():
    """
    This takes the local file credentials.yaml, converts it to a dictionary and returns this dictionary.
    
    """
    with open('credentials.yaml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data

class RDSDatabaseConnector:
    """
    An object of this class is initialised with a dictionary of the credentials of the RDS database that
    it is connecting to.

    The init_sql_engine() method takes the dictionary of credentials and uses them to create an engine for 
    the database which can be used to make connections and queries. It then assigns this engine to a new 
    attribute of the object, self.engine, because each engine is unique to each RDSDatabaseConnector.

    The extract_loan_payments_data() method is applied after the init_sql_engine() method because it uses
    the engine created by that method to make a specific query: extract the loan_payments table. It then
    returns this table as a Pandas dataframe.
    """
    def __init__(self, dict_credentials):
        self.dict_credentials = dict(dict_credentials)

    def init_sql_engine(self):
        db_user = self.dict_credentials['RDS_USER']
        db_password = self.dict_credentials['RDS_PASSWORD']
        db_host = self.dict_credentials['RDS_HOST']
        db_port = self.dict_credentials['RDS_PORT']
        db_name = self.dict_credentials['RDS_DATABASE']
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return create_engine(db_url)

class Extractor:
    def __init__(self):
        pass

    def get_table_as_df(self, engine, table):
        sql_query = f'SELECT * FROM {table}'
        df = pd.read_sql_query(sql_query, engine)
        return df

class CSV_DF:
    def __init__(self):
        pass

    def save_as_csv(self, df, file_name):
        """
        This function takes a Pandas df and a file_name, and saves the df as a csv file locally, named file_name.

        The '.csv' needs to be included in the file_name specified.
        """
        df.to_csv(f'{file_name}', index = False)
        
    def read_csv_to_df(self, file_path):
        """
        This function takes a locally saved csv file, converts it into a Pandas df and returns it.
        """
        df = pd.read_csv(file_path)
        return df

class DataTransformer:
    def __init__(self):
        pass

    def convert_datestrings_to_iso(self, df, column_name):
        """
        Convert date strings in a DataFrame column from the format "MMM-YYYY" to ISO format "MM-YYYY" in place.

        Parameters:
        - df: pandas DataFrame
            The DataFrame containing the date column.
        - column_name: str
            The name of the column containing date strings in the format "MMM-YYYY".
        """
        # Function to convert date string to ISO format
        def convert_datestring_to_iso(datestring):
            if pd.isnull(datestring):
                date_object = None
            else:
                try:
                    date_object = datetime.strptime(datestring, '%b-%Y').strftime('%m-%Y')
                except:
                    date_object = datestring
            return date_object

        # Apply the conversion function to the column
        df[column_name] = df[column_name].apply(convert_datestring_to_iso)

    def convert_to_int64(self, dataframe, column_name):
        dataframe[column_name] = dataframe[column_name].astype('Int64')

    def convert_to_float64(self, dataframe, column_name):
        dataframe[column_name] = dataframe[column_name].astype('Float64')

    def convert_yearstrings_to_float64(self, dataframe, column_name):
        def yearstring_to_float64(yearstring):
            numeric_part = ''.join(yearstring.split()[0])
            return numeric_part
        dataframe[column_name] = dataframe[column_name].apply(yearstring_to_float64)

    def convert_monthstrings_to_float64(self, dataframe, column_name):
        def monthstring_to_float64(monthstring):
            numeric_part = ''.join(monthstring.split()[:1])
            return float(numeric_part)
        dataframe[column_name] = dataframe[column_name].apply(monthstring_to_float64)
    
    
    def drop_column(self, dataframe, column_name):
        dataframe.drop(column_name, axis=1, inplace=True)

    def impute_mean(self, dataframe, column_name):
        dataframe[column_name] = dataframe[column_name].fillna(dataframe[column_name].mean())

    def impute_median(self, dataframe, column_name):
        dataframe[column_name] = dataframe[column_name].fillna(dataframe[column_name].median())

    def impute_mode(self, dataframe, column_name):
        dataframe[column_name] = dataframe[column_name].fillna(dataframe[column_name].mode().iloc[0])

    def log_transformation(self, dataframe, column_name):
        dataframe[column_name] = dataframe[column_name].map(lambda i: np.log(i) if i > 0 else 0)

    def boxcox_transformation(self, dataframe, column_name):
        boxcox_d = dataframe[column_name]
        boxcox_d = stats.boxcox(boxcox_d)
        dataframe[column_name] = pd.Series(boxcox_d[0])

    def yeojohnson_transformation(self, dataframe, column_name):
        yeojohnson_d = dataframe[column_name]
        yeojohnson_d = stats.yeojohnson(yeojohnson_d)
        dataframe[column_name] = pd.Series(yeojohnson_d[0])

class Plotter:
    def __init__(self):
        pass

    def qq_plot(self, dataframe, column_name):
        qq_plot = qqplot(dataframe[column_name] , scale=1 ,line='q')
        pyplot.show()

    def hist_plot_skewness(self, dataframe, column_name, bin = 40):
        t=sns.histplot(dataframe[column_name],label="Skewness: %.2f"%(dataframe[column_name]).skew(), bins=bin)
        t.legend()

    def boxplot_all_float_cols(self, dataframe):
        df = dataframe.select_dtypes(include='float64')
        for column in df:
            pyplot.figure(figsize=(6, 4))
            sns.boxplot(data=df[column])
            pyplot.title(f'Box plot for column {column}')
            pyplot.ylabel('Values')
            pyplot.show()

    def histplot_all_float_cols(self, dataframe):
        df = dataframe.select_dtypes(include='float64')
        for column in df:
            pyplot.figure(figsize=(6, 4))
            sns.histplot(data=df[column], bins = 40, kde=False)
            pyplot.title(f'Box plot for column {column}')
            pyplot.ylabel('Values')
            pyplot.show()

    def plot_corr_matrix(self, dataframe):
        px.imshow(dataframe.corr(), title="Correlation heatmap of student dataframe")


class DataFrameInfo:
    def __init__(self):
        pass

    def compare_mean_median(self, dataframe, column_name):
        print(f'Mean: {dataframe[column_name].mean()}')
        print(f'Median: {dataframe[column_name].median()}')

    def check_null_percentages(self, dataframe):
        print((dataframe.isnull().sum()/len(loan_payments))*100)

    def get_skews(self, dataframe):
        skewed_data = []
        for col in list(dataframe.columns):
            try:
                print(f'Skew of {col}: {dataframe[col].skew()}')
                if abs(dataframe[col].skew()) > 4:
                    skewed_data.append(col)
                else:
                    pass

            except:
                pass

        print(f'Columns with absolute skew values greater than 4: {skewed_data}')
        return skewed_data
    
    def get_numeric_columns(self, dataframe):
        numeric_df = dataframe.select_dtypes(include='number')
        return numeric_df




#%% Setting up connection, creating engine and inspecting the data

connector_1 = RDSDatabaseConnector(credentials_to_dict())
engine_1 = connector_1.init_sql_engine()
extractor_1 = Extractor()
loan_payments = extractor_1.get_table_as_df(engine=engine_1, table='loan_payments')
loan_payments.head()
# %% Changing all the date-based columns to datetime ISO format
datestring_cols = ['issue_date','earliest_credit_line','last_payment_date', 'last_credit_pull_date']
transformer_1 = DataTransformer()
for column in datestring_cols:
    transformer_1.convert_datestrings_to_iso(loan_payments, column)
#%% Changing two columns from float64 to int 64
transformer_1 = DataTransformer()
transformer_1.convert_to_int64(loan_payments, 'mths_since_last_delinq')
transformer_1.convert_to_int64(loan_payments, 'mths_since_last_record')
transformer_1.convert_to_int64(loan_payments, 'mths_since_last_major_derog')


# %% Saving downloaded dataframe locally
csv_df_1 = CSV_DF()
csv_df_1.save_as_csv(loan_payments, 'loan_payments.csv')

# %% Dropping columns with a high percentage of missing values
dfinfo_1 = DataFrameInfo()
dfinfo_1.check_null_percentages(loan_payments)
transformer_1.drop_column(loan_payments, 'mths_since_last_delinq')
transformer_1.drop_column(loan_payments, 'mths_since_last_record')
transformer_1.drop_column(loan_payments, 'next_payment_date')
transformer_1.drop_column(loan_payments, 'mths_since_last_major_derog')

# %% Histplots for all cols of dtype 'float64' to do visual inspection, looking for skew and for whether any float64 data needs to be converted to int64
plotter_1 = Plotter()
plotter_1.histplot_all_float_cols(loan_payments)

#%% Zooming in on one column that might need to be converted to int64
loan_payments['collections_12_mths_ex_med'].value_counts()
#%% This needs to be converted from 'float64' to 'int64'
transformer_1 = DataTransformer()
transformer_1.impute_mode(loan_payments, 'collections_12_mths_ex_med')
transformer_1.convert_to_int64(loan_payments, 'collections_12_mths_ex_med')

#%% Identifying which numeric columns are skewed and correcting skew using a Yeo-Johson transformation
dfinfo_1 = DataFrameInfo()
plotter_1 = Plotter()
transformer_1 = DataTransformer()
skewed_data = dfinfo_1.get_skews(loan_payments)

for i in range(0, len(skewed_data)-1):
    transformer_1.yeojohnson_transformation(loan_payments, str(skewed_data[i]))
# Could in theory make this bit of code a method of DataTransformer class - ??

#%% Boxplots to look for outliers
plotter_1 = Plotter()
plotter_1.boxplot_all_float_cols(loan_payments)
# No obvious ones identified - ??
#%%
transformer_1 = DataTransformer()
transformer_1.impute_mode(loan_payments, 'term')
transformer_1.convert_monthstrings_to_float64(loan_payments, 'term')

# %% Dropping highly corrleated columns
numeric_df = loan_payments.select_dtypes(include='number')
px.imshow(numeric_df.corr(), title="Correlation heatmap of student dataframe")
#%% If column correlation > 0.9 drop one of them
transformer_1 = DataTransformer()
transformer_1.drop_column(numeric_df, 'funded_amount_inv')
transformer_1.drop_column(numeric_df, 'out_prncp_inv')
transformer_1.drop_column(numeric_df, 'total_payment_inv')
# %%
px.imshow(numeric_df.corr(), title="Correlation heatmap of student dataframe")
# %% Couple more to drop
transformer_1 = DataTransformer()
transformer_1.drop_column(numeric_df, 'total_rec_prncp')
transformer_1.drop_column(numeric_df, 'funded_amount')
# %%