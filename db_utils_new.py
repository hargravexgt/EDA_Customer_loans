

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

class DataTransform:
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

    def convert_yearstrings_to_float64_2(self, df, column_name):
        
        def yearstring_to_float64(yearstring):
            try:
                numeric_part = yearstring.split('y')[0].strip()

                if numeric_part == '< 1':
                    numeric_part = 0.0
                elif numeric_part == '10+':
                    numeric_part = 10.0
                else:
                    numeric_part = float(numeric_part)

            except AttributeError:
                numeric_part = None

            return numeric_part
        
        df[column_name] = df[column_name].apply(yearstring_to_float64)

    def convert_monthstrings_to_float64(self, dataframe, column_name):
        def monthstring_to_float64(monthstring):
            numeric_part = ''.join(monthstring.split()[:1])
            return float(numeric_part)
        dataframe[column_name] = dataframe[column_name].apply(monthstring_to_float64)

class DataFrameTransform:
    def __init__(self):
        pass

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

    def PDF_plot_with_averages(self, dataframe, column_name):

        try:
            sns.histplot(dataframe[column_name], kde=True, stat='density')

            pyplot.xlabel(f'Values of {column_name}')
            pyplot.ylabel('Probability')
            pyplot.title('Probability Density Function')
            pyplot.show()
            print(f"The mode of the distribution is {dataframe[column_name].mode()[0]}")
            print(f"The mean of the distribution is {dataframe[column_name].mean()}")
            print(f"The median of the distribution is {dataframe[column_name].median()}")

        except TypeError:
            raise TypeError('Not numeric data so cannot be plotted with PDF')

class DataFrameInfo:
    def __init__(self):
        pass

    def compare_mean_median(self, dataframe, column_name):
        print(f'Mean: {dataframe[column_name].mean()}')
        print(f'Median: {dataframe[column_name].median()}')

    def check_null_percentages(self, dataframe):
        print((dataframe.isnull().sum()/len(dataframe))*100)

    def check_no_of_nulls(self, dataframe):
        print(dataframe.isnull().sum())

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
