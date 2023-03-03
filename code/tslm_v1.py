import re
import pandas as pd
import numpy as np
from beartype import beartype
from sklearn.linear_model import LinearRegression

def set_index(data:pd.DataFrame, date_col:str) -> pd.DataFrame:
    # 데이터 카피Quarter
    data_ = data.copy()
    # 인덱스 설정
    data_.index = pd.PeriodIndex(data_[date_col].str.replace(' Q','-Q', regex=True), freq='Q').to_timestamp()
    # date컬럼 제거
    data_.drop(columns=date_col, inplace=True)
    
    return data_

@beartype
def mk_train_mart(y:pd.DataFrame, x:pd.DataFrame = None, trend:bool = None, freq:int = None) -> pd.DataFrame:
    # argument 확인
    # y 확인: 빈 데이터프레임인지 확인
    if y.empty:
        raise ValueError("'y' is an empty DataFrame.")
        
    # x 확인: datetime 인덱스인지 확인
    if not isinstance(x, pd.DataFrame):
        data_ = pd.DataFrame()
    else:
        if isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex):
            x = x.sort_index()
            data_ = x.copy()
        else:
            raise ValueError("'x' index is not DatetimeIndex.")
    
    # x, y 확인: 로우 수가 같은지 확인
    if not data_.empty and len(y) != len(data_):
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [y:{len(y)}, x:{len(data_)}]")

    # trend 컬럼 생성
    if trend:
        data_['trend'] = y.reset_index().index + 1
        
    # Season 컬럼 생성
    if freq:
        data_['season'] = data_['trend'].apply(lambda x: freq if x % freq == 0 else x % freq)
        season_cols = pd.get_dummies(data_['season'], prefix='season')
        data_ = pd.concat([data_, season_cols], axis=1).drop(columns=['season', 'season_1'])
    return data_

@beartype
def fit(x:pd.DataFrame, y:pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    return model.fit(x, y)

@beartype
def summary_model(fit_obj:LinearRegression) -> pd.DataFrame:
    coef = pd.DataFrame(fit_obj.coef_, columns=fit_obj.feature_names_in_)
    intercept = pd.DataFrame(fit_obj.intercept_, columns=['(intercept)'])
    summary = pd.concat([intercept, coef], axis=1).T
    summary.rename(columns={0:'value'}, inplace=True)
    return summary.reset_index()

@beartype
def tslm(y:pd.DataFrame, x:pd.DataFrame = None, trend:bool = None, freq:int = None) -> LinearRegression:
    # step1. tslm 학습 마트 생성
    train_mart = mk_train_mart(y=y, x=x, trend=trend, freq=freq)
    
    # step2. fitting
    fit_obj = fit(x=train_mart, y=y)
    
    return fit_obj

@beartype
def predict(fit_obj:LinearRegression, train_mart:pd.DataFrame, new_data:pd.DataFrame = None, 
            trend:bool = None, freq:int = None, h:int=None) -> np.ndarray:
    # argument 확인
    # train_mart 확인: 빈 데이터프레임인지 확인
    if train_mart.empty:
        raise ValueError("'new_data' is an empty DataFrame.")
    
    # new_data 및 h 확인
    features = [f for f in fit_obj.feature_names_in_ if not re.match('(trend)|(^season_)', f)]
    
    if isinstance(new_data, pd.DataFrame):
        if new_data.empty and len(features) !=0:
            raise ValueError(f"'new_data' is empty.")
    else:
        if len(features) !=0:
            raise ValueError(f"'new_data' is none.")
        elif h == None:
            raise ValueError("both 'new_data' and 'h' are None.")
    
    # 데이터 복사 및 생성
    data_ = pd.DataFrame() if new_data.empty else new_data.copy()
    
    # trend 컬럼 생성
    if trend:
        last_trend = len(train_mart)
        pred_count = h if new_data.empty else len(data_)
        trend_values = map(lambda x: x + last_trend + 1, range(1, pred_count+1))
        data_['trend'] = list(trend_values)
        
    # Season 컬럼 생성
    if freq:
        data_['season'] = data_['trend'].apply(lambda x: freq if x % freq == 0 else x % freq)
        season_cols = pd.get_dummies(data_['season'], prefix='season')
        data_ = pd.concat([data_, season_cols], axis=1).drop(columns=['season', 'season_1'])
    return fit_obj.predict(data_)


# if __name__ == '__main__':
    # raw = pd.read_csv('https://raw.githubusercontent.com/hyesinn/tspkg/main/Data/aus_production.csv')
    # data = set_index(raw, 'Quarter')
    # train_data = data[data.index < '2009-01-01']
    # test_data = data[data.index >= '2009-01-01']
    # trian_mart = mk_train_mart(y=train_data[['Beer']], x=train_data[['Cement']], trend=True, freq=4)
    # tslm_model = tslm(y=train_data[['Beer']], x=train_data[['Cement']], trend=True, freq=4)
    # pred = predict(fit_obj=tslm_model, train_mart=trian_mart, new_data = test_data[['Cement']], trend=True, freq=4)
    # print(pred)
