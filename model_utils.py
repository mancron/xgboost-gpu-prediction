# model_utils.py
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from db_config import get_engine


def load_data():
    engine = get_engine()
    query = "SELECT name, date, avg_price, min_price, max_price, std_dev FROM ref_vga_stats"
    df = pd.read_sql(query, engine)
    return df



def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['name', 'date'])

    # 이상치 제거 (Z-score 기준)
    def remove_outliers(group):
        z_scores = (group['avg_price'] - group['avg_price'].mean()) / group['avg_price'].std()
        return group[(z_scores > -3) & (z_scores < 3)]

    df = df.groupby('name', group_keys=False).apply(remove_outliers)

    # 로그 변환 (로그1p = log(x + 1), 0 이상에서 안정적)
    df['avg_price'] = np.log1p(df['avg_price'])
    df['min_price'] = np.log1p(df['min_price'])
    df['max_price'] = np.log1p(df['max_price'])

    df['avg_price_pct_change'] = df.groupby('name')['avg_price'].pct_change()
    df['avg_price_ma7'] = df.groupby('name')['avg_price'].transform(lambda x: x.rolling(7).mean())
    df['avg_price_ma30'] = df.groupby('name')['avg_price'].transform(lambda x: x.rolling(30).mean())

    # 타깃 변수도 로그 적용한 avg_price 기준으로 설정
    df['target'] = df.groupby('name')['avg_price'].shift(-1)

    df = df.dropna()
    return df


def train_model(df, gpu_name):
    df_gpu = df[df['name'] == gpu_name]
    if len(df_gpu) < 50:
        return None, None

    # min_price, max_price 제거
    feature_cols = ['avg_price', 'std_dev',
                    'avg_price_pct_change', 'avg_price_ma7', 'avg_price_ma30']
    X = df_gpu[feature_cols]
    y_log = df_gpu['target']

    # 로그 값을 실제 가격으로 변환 (expm1)해서 실제 가격 기준으로 RMSE 계산
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, shuffle=False)
    y_test_real = np.expm1(y_test_log)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    return model, df_gpu.iloc[-1:][feature_cols], rmse

def predict_price(model, latest_row):
    log_pred = model.predict(latest_row)[0]
    return np.expm1(log_pred)  # 로그 복원
