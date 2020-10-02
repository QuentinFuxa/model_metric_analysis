from scipy.stats import linregress
import numpy as np
import pandas as pd
import warnings
from IPython.display import clear_output
import pickle
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import logging
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

logging.basicConfig(
         format='%(asctime)s  [%(funcName)s] %(message)s',
         level=1,
         datefmt='%Y-%m-%d %H:%M:%S')

def reset_index(df):
    if df.index.name != None or df.index.names[0] != None:
        df.reset_index(inplace=True)
    return df

def result_per_key(
        df, 
        key , 
        groupby_col, 
        pred_col = 'pred', 
        y_col = 'actual',
        minimal = True, 
        type_p = int, 
        l_index = False
    ):
    """
    Adapted call to the regression_results function in case of multiple models
    :param df: DataFrame containing the results of the models and the actual quantities.
    :param by_key: int/str/list containing the desired output granularity
    :param pred_col: column containing the results of the machine learning model.
    :param y_col: column containing the actual quantity
    :param extend_key: key on which we want to see the sum difference between pred and actual
    :param minimal: print more metrics
    :param extend: when objective for a sum, print the difference
    :param type_p: type of data. For inbound modules: int, for bounds: float
    :param l_index: adapt the index name if by_key contains a list
    :return: DataFrame containing the error
    """
    df = reset_index(df)
    df = df.set_index(key)
    result_plot = pd.DataFrame()
    for index in df.index.unique():
        sub_df = df.loc[index]
        result_sub_df = regression_results_df(sub_df, groupby_col, pred_col, y_col, row = index, minimal = minimal,
                                            type_p = type_p)
        if l_index:
            result_sub_df.index = [str(result_sub_df.index[0][0]) + '-' + result_sub_df.index[0][1]]
        result_plot = result_plot.append(result_sub_df)
    return result_plot.sort_values('sum actual', ascending = False)



def regression_results_df(
        df, 
        groupby_col, 
        pred_col, 
        y_col, row = '', 
        minimal = True, 
        type_p = int
    ):
    """
    Produces a DataFrame of 1 row containing different error metrics between prediction & actual inside a DataFram of 1 row and several columns
    :param df: DataFrale
    :param groupby_col: granularity used to compare the pred and actual sums.
    :param pred_col: column containing the predictions
    :param y_col: column containing the actual quantities
    :param row: name of the row
    :param minimal: print more metrics
    :param type_p: type of data. For inbound modules: int, for bounds: float
    :return: DataFrame containing the error
    """

    y_true = df[y_col]
    y_pred = df[pred_col]
    mean_absolute_error= np.abs(metrics.mean_absolute_error(y_true, y_pred))
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    sum_actual = df.groupby(groupby_col)[y_col].sum()
    sum_pred = df.groupby(groupby_col)[pred_col].sum()
    diff = np.mean((sum_pred - sum_actual)/sum_actual)

    result_dict = {}
    result_dict['r2'] = float(round(r2,4))
    result_dict['MAE/mean'] = float(round(mean_absolute_error/np.mean(y_true),4))
    result_dict['RMSE'] = round(np.sqrt(mse),4)
    result_dict['mean'] = round(np.mean(y_true),4)
    result_dict['mean pred'] = round(np.mean(y_pred),4)
    result_dict['std'] = np.sqrt(np.var(y_true))
    result_dict['nb. obs'] = len(y_true)
    result_dict['nb. positive obs'] = sum(y_true > 0)
    result_dict['sum actual'] = int(sum(y_true))
    result_dict['% diff.'] = round(diff,3)*100

    if not minimal:
        slope, intercept, r_value, p_value, std_err = linregress(x=y_true, y=y_pred)
        explained_variance=metrics.explained_variance_score(y_true, y_pred)
        median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        result_dict['MAE'] = round(mean_absolute_error,4)
        result_dict['MSE'] = round(mse,4)
        result_dict['slope'] = round(slope,4)
        result_dict['slope'] = round(slope,4)
        result_dict['intercept'] =  round(intercept, 4)
        result_dict['str_err'] = round(std_err,4)

    df = pd.DataFrame(result_dict, index = [row]).sort_index()
    df[['RMSE', 'mean', 'mean pred', 'std', 'nb. obs', 'nb. positive obs', 'sum actual']] = \
        df[['RMSE', 'mean', 'mean pred', 'std', 'nb. obs', 'nb. positive obs', 'sum actual']].astype(type_p)
    if not minimal:
        df[['MAE', 'MSE']] = df[['MAE', 'MSE']].astype(type_p)
    return df


def regression_results(
        y_pred, 
        y_true, 
        row = '', 
        minimal = True, 
        type_p = int
    ):
    """
    Produces a DataFrame of 1 row containing different error metrics between prediction & actual inside a DataFram of 1 row and several columns
    :param y_pred: column containing the predictions
    :param y_true: column containing the actual quantities
    :param row: name of the row
    :param minimal: print more metrics
    :param type_p: type of data. For inbound modules: int, for bounds: float
    :return: DataFrame containing the error
    """
    mean_absolute_error= np.abs(metrics.mean_absolute_error(y_true, y_pred))
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    result_dict = {}
    result_dict['r2'] = float(round(r2,4))
    result_dict['MAE/mean'] = float(round(mean_absolute_error/np.mean(y_true),4))
    result_dict['RMSE'] = round(np.sqrt(mse),4)
    result_dict['mean'] = round(np.mean(y_true),4)
    result_dict['mean pred'] = round(np.mean(y_pred),4)
    result_dict['std'] = np.sqrt(np.var(y_true))
    result_dict['nb. obs'] = len(y_true)
    result_dict['nb. positive obs'] = sum(y_true > 0)
    result_dict['sum actual'] = int(sum(y_true))
    if not minimal:
        slope, intercept, r_value, p_value, std_err = linregress(x=y_true, y=y_pred)
        explained_variance=metrics.explained_variance_score(y_true, y_pred)
        median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        result_dict['MAE'] = round(mean_absolute_error,4)
        result_dict['MSE'] = round(mse,4)
        result_dict['slope'] = round(slope,4)
        result_dict['slope'] = round(slope,4)
        result_dict['intercept'] =  round(intercept, 4)
        result_dict['str_err'] = round(std_err,4)
    df = pd.DataFrame(result_dict, index = [row]).sort_index()
    df[['RMSE', 'mean', 'mean pred', 'std', 'nb. obs', 'nb. positive obs', 'sum actual']] = \
        df[['RMSE', 'mean', 'mean pred', 'std', 'nb. obs', 'nb. positive obs', 'sum actual']].astype(type_p)
    if not minimal:
        df[['MAE', 'MSE']] = df[['MAE', 'MSE']].astype(type_p)
    return df
