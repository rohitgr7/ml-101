#### TODO ####
# Correct the docs
# Make fix_missing optimal
# Try adding day and dayofyear in attr of add_datecols()


import numpy as np
import pandas as pd
import re
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklear.ensemble import forest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class LabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = ['NaN'] + list(set(y) - set(['NaN']))
        self.class_maps_ = {label: i for i, label in enumerate(self.classes_)}
        self.inverse_maps_ = {i: label for label, i in self.class_maps_.items()}

        return self

    def transform(self, y):
        y = np.array(y)
        new_labels = list(set(y) - set(self.classes_))
        y[np.isin(y, new_labels)] = 'NaN'

        return np.array([self.class_maps_[v] for v in y]).astype(np.int32)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        y = np.array(y)
        return np.array([self.inverse_maps_[v] for v in y])

    def add_label(self, label, ix):
        self.classes_ = self.classes_[:ix] + [label] + self.classes_[ix:]
        self.class_maps_ = {label: i for i, label in enumerate(self.classes_)}
        self.inverse_maps_ = {i: label for label, i in self.class_maps_.items()}


def replace_nan(df, nan_cols):
    df.replace(nan_cols, np.nan, inplace=True)


def tv_split(X, y, valid_sz=0.2, temporal=None, optim=False):
    """
    Split data into train and validation data

    Parameters
    ----------
        X: Independant data
        y: Dependant data
        valid_sz: Size of validation set b/w 0 and 1
        temporal: (Boolean) Wheather the data has temporal column or not
        optim: (Boolean) Return optimized train and validation set if true and data is not temporal

    Returns
    -------
        X: Independant data
        y: Dependant data
        na_dict: Dictionary mapping for columns with missing data
        mapper: A DataFrameMapper which stores mean and standard deviation for continuous data which is then used for scaling
    """
    shuffle = True
    stratify = None

    if optim:
        stratify = y

    if temporal is not None:
        sort_idx = np.argsort(X[temporal].values)
        X, y = X[sort_idx], y[sort_idx]
        shuffle = False

    return train_test_split(X, y, test_size=valid_sz, shuffle=shuffle, stratify=stratify)


def conv_contncat(df, cont_cols=None, cat_cols=None):
    if cont_cols is not None:
        for n in cont_cols:
            df[n] = pd.to_numeric(df[n], errors='coerce').astype(np.float64)

    if cat_cols is not None:
        for n in cat_cols:
            df[n] = df[n].astype(str)

    df[df == 'nan'] = np.nan


def add_datecols(df, col, time=True, drop=True):
    """
    Add date-related columns

    Parameters
    ----------
        df: Dataframe to be processed
        col: Datetime column name
        time: (Boolean) Wheather to add time related columns or not
        drop: (Boolean) Wheather to drop the original column or not

    Returns
    -------
        None
    """

    fld_dtype = df[col].dtype

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

    target_pre = re.sub('[Dd]ate$', '', col)
    attr = ['year', 'month', 'week', 'dayofweek', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', ]

    if time:
        attr += ['hour', 'minute', 'second']

    for at in attr:
        df[target_pre + at + '_cat'] = getattr(df[col].dt, at)

    df[target_pre + 'elapsed'] = np.int64(df[col])

    if drop:
        df.drop(col, axis=1, inplace=True)


def get_nn_mappers(df, cat_cols, cont_cols):
    """
    Create and return mappers for categorical and continuous data

    Parameters
    ----------
        df: Dataframe to be processed
        cat_cols: Datetime column name
        cont_cols: (Boolean) Wheather to add time related columns or not

    Returns
    -------
        CategoricalMapper, ContinuousMapper
    """

    cat_maps = [(o, LabelEncoder()) for o in cat_cols]
    cont_maps = [([o], StandardScaler()) for o in cont_cols]

    conv_mapper = DataFrameMapper(cont_maps).fit(df)
    cat_mapper = DataFrameMapper(cat_maps).fit(df)
    return cat_mapper, conv_mapper


def fix_missing(df, na_dict=None, cont_cols=None, cat_cols=None):
    """
    Fill continous mssing values with mean and categorical missing values with 'NaN'

    Parameters
    ----------
        df: Dataframe to be processed
        na_dict: Mapping of columns and values to be filled in the missing place corresponding to the column

    Returns
    -------
        na_dict: A dictionary containg the map of column names with the corresponding value to be filled in missing place
    """

    if na_dict is not None:
        for n in na_dict.keys():
            col_null = df[n].isnull()
            df[n + '_na'] = col_null
            df.loc[col_null, n] = na_dict[n]

        if cont_cols is not None:
            for n in cont_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df.loc[col_null, n] = df[n].median()

        if cat_cols is not None:
            for n in cat_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df.loc[col_null, n] = 'NaN'

    else:
        na_dict = {}
        if cont_cols is not None:
            for n in cont_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df[n + '_na'] = col_null
                    na_dict[n] = df[n].median()
                    df.loc[col_null, n] = na_dict[n]

        if cat_cols is not None:
            for n in cat_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df[n + '_na'] = col_null
                    na_dict[n] = 'NaN'
                    df.loc[col_null, n] = na_dict[n]

    return na_dict


def get_one_hot(df, max_n_cat, drop_first, cat_mapper):
    one_hot_cols = []

    for n, c in df.items():
        if n.endswith('_cat') and len(set(df[n])) <= max_n_cat:
            one_hot_cols.append(n)

    for n, encoder, _ in cat_mapper.built_features:
        if len(encoder.classes_) <= max_n_cat:
            one_hot_cols.append(n)

    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=drop_first)
    return df


def proc_df(df, y_col, subset=None, drop_cols=None, do_scale=False, cat_mapper=None, cont_mapper=None, max_n_cat=None, drop_first=False):
    """
    Split data into x, y

    Parameters
    ----------
        df: Dataframe to be processed
        y_col: Column name of the df to be predicted
        subset: dtype(int) Number of rows required from the data
        drop_cols: Columns to be dropped
        na_dict: A dictionary mapping column to missing values to be filled
        do_scale: (Boolean) Wheather to scale the independant value or not
        mapper: A DataFrameMapper containg mean and std of columns used to normalize the data
        max_n_cat: Minimum number of categories to create one-hot encodings
        drop_first: (Boolean) Wheather to drop first column during one_hot_encoding

    Returns
    -------
        X: Independant data
        y: Dependant data
        na_dict: dictionarynary mapping for columns with missing data
        mapper: A DataFrameMapper which stores mean and standard deviation for continuous data which is then used for scaling
    """

    if subset is not None:
        df = df[-subset:]

    if drop_cols is not None:
        df.drop(drop_cols, axis=1, inplace=True)

    if do_scale:
        df[cont_mapper.transformed_names_] = cont_mapper.transform(df)

    df[cat_mapper.transformed_names_] = cat_mapper.transform(df)

    if max_n_cat is not None:
        df = get_one_hot(df, max_n_cat, drop_first, cat_mapper)

    X = df.drop(y_col, axis=1)
    y = df[y_col]

    return X, y


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}).sort_values('imp', ascending=False)


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))


def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n_samples))
