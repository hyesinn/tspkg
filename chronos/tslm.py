import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from abc import ABCMeta, abstractmethod


class NotFittedError(Exception):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility."""


class TimeSerieseModel(object, metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        '''Fit model'''
    
    @abstractmethod
    def predict(self):
        '''Predict'''
        
class TSLM(TimeSerieseModel):
    def __init__(self, trend:bool=None, freq:int=None):
        self.trend = trend
        self.freq = freq
        
    def __repr__(self):
        class_name = type(self).__name__        
        return '{}(trend={}, freq={})'.format(class_name, self.trend, self.freq)
        
    def __check_fit_argus(self):
        # argument 확인
        # target 확인: 빈 데이터프레임인지 확인
        if self.target.empty:
            raise ValueError("'target' is an empty DataFrame.")

        # x 확인: datetime 인덱스인지 확인
        if not isinstance(self.raw, pd.DataFrame):
            self.raw = pd.DataFrame()
        else:
            if isinstance(self.raw.index, pd.core.indexes.datetimes.DatetimeIndex):
                self.raw = self.raw.sort_index()
            else:
                raise ValueError("'x' index is not DatetimeIndex.")

        # x, y 확인: 로우 수가 같은지 확인
        if not self.raw.empty and len(self.target) != len(self.raw):
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [y:{len(self.target)}, x:{len(self.raw)}]")
            
        if len(self.target) < self.freq*2:
            raise ValueError("the train data size is smaller than 'freq'.")
            
    def __check_fitted(self):
        if not hasattr(self, '_TSLM__model'):
            error_msg = f"This {type(self).__name__} instance is not fitted yet."
            raise NotFittedError(error_msg)
    
    def __check_pred_argus(self):        
        if isinstance(self.__new_data, pd.DataFrame):
            if self.__new_data.empty and len(self.raw) != 0:
                raise ValueError(f"'new_data' is empty.")
            elif len(self.__new_data) > 0 and len(self.raw) == 0:
                raise ValueError(f"'new_data' is not empty.")
        else:
            if len(self.raw) !=0:
                raise ValueError(f"'new_data' is none.")
            elif self.__h == None:
                raise ValueError("both 'new_data' and 'h' are None.")

    def mk_ts_mart(self, cnt_trend):
        ts_mart = pd.DataFrame()
        
        if self.trend:
            ts_mart['trend'] = range(1, cnt_trend+1)

        if self.freq:
            tmp_season = ts_mart['trend'].apply(lambda x: self.freq if x % self.freq == 0 else x % self.freq)
            ts_mart['season'] = pd.Categorical(tmp_season, range(1, self.freq+1))
            season_cols = pd.get_dummies(ts_mart['season'], prefix='season')
            ts_mart = pd.concat([ts_mart, season_cols], axis=1).drop(columns=['season', 'season_1'])
        return ts_mart
    
    def fit(self, y:pd.DataFrame, x:pd.DataFrame=None):
        self.target = y
        self.raw = x
        TSLM.__check_fit_argus(self)        
        self.ts_mart = TSLM.mk_ts_mart(self, cnt_trend=len(self.target))
        self.ts_mart.index = self.target.index
        self.train_mart = pd.concat([self.raw, self.ts_mart], axis=1)
        
        model = LinearRegression()
        self.__model = model.fit(self.train_mart, self.target)
        return self
    
    def summary_model(self):
        TSLM.__check_fitted(self)
        coef = pd.DataFrame(self.__model.coef_, columns=self.__model.feature_names_in_)
        intercept = pd.DataFrame([self.__model.intercept_], columns=['(intercept)'])
        summary = pd.concat([intercept, coef], axis=1).T
        summary.rename(columns={0:'value'}, inplace=True)
        return summary.reset_index()
    
    def predict(self, new_data:pd.DataFrame=None, h:int=None):
        self.__new_data = new_data
        self.__h = h
        TSLM.__check_fitted(self)
        TSLM.__check_pred_argus(self)
        
        if h == None:
            cnt_trend = len(new_data)
            pred_data = new_data.reset_index(drop=True)
        else:
            cnt_trend = h
            pred_data = pd.DataFrame()
         
        ts_mart = TSLM.mk_ts_mart(self, cnt_trend= cnt_trend)
        ts_mart['trend'] = ts_mart['trend'] + len(self.train_mart) + 1
        fin_pred_mart = pd.concat([pred_data, ts_mart], axis=1)
        
        pred = self.__model.predict(fin_pred_mart)
        
        return pred