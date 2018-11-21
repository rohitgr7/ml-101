#### TODO ####
# Correct the docs
# Make fix_missing optimal
# Try adding day and dayofyear in attr of add_datecols()
# fix tv_split


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


def get_nn_mappers(df, cat_cols=None, cont_cols=None):
    cat_mapper, cont_mapper = None, None
    
    if cat_cols is not None:
        cat_maps = [(o, LabelEncoder()) for o in cat_cols]
        cat_mapper = DataFrameMapper(cat_maps).fit(df)
    
    if cont_cols:
        cont_maps = [([o], StandardScaler()) for o in cont_cols]
        cont_mapper = DataFrameMapper(cont_maps).fit(df)
    
    return cat_mapper, cont_mapper


def fix_missing(df, na_dict=None, cont_cols=None, cat_cols=None, add_na=True):
    if na_dict is not None:
        for n in na_dict.keys():
            col_null = df[n].isnull()
            if add_na:
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
                    if add_na:
                        df[n + '_na'] = col_null
                    na_dict[n] = df[n].median()
                    df.loc[col_null, n] = na_dict[n]

        if cat_cols is not None:
            for n in cat_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    if add_na:
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


def proc_df(df, y_col=None, subset=None, drop_cols=None, do_scale=False, cat_mapper=None, cont_mapper=None, max_n_cat=None, drop_first=False):
    df = df.copy()
    
    if subset is not None:
        df = df[-subset:]

    if drop_cols is not None:
        df.drop(drop_cols, axis=1, inplace=True)

    if do_scale:
        df[cont_mapper.transformed_names_] = cont_mapper.transform(df)
    
    if cat_mapper:
        df[cat_mapper.transformed_names_] = cat_mapper.transform(df)

    if max_n_cat is not None:
        df = get_one_hot(df, max_n_cat, drop_first, cat_mapper)

    if y_col is not None:
        X = df.drop(y_col, axis=1)
        y = df[y_col]
        return X, y

    return df


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}).sort_values('imp', ascending=False)


def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))


def reset_rf_samples():
	forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n_samples
