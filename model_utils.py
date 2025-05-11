import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from db_config import get_engine


# 데이터 로드 함수
def load_data():
    engine = get_engine()
    query = "SELECT name, date, avg_price, std_dev FROM ref_vga_stats"
    df = pd.read_sql(query, engine)
    return df


# 데이터 전처리 함수
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
    df['std_dev'] = np.log1p(df['std_dev'])

    # 타깃 변수도 로그 적용한 avg_price 기준으로 설정
    df['target'] = df.groupby('name')['avg_price'].shift(-1)

    df = df.dropna()
    return df


# GPU 가격 예측 모델 학습
def train_model(df, gpu_name):
    df_gpu = df[df['name'] == gpu_name]
    if len(df_gpu) < 50:
        return None, None, "데이터가 부족하여 모델을 학습할 수 없습니다."

    feature_cols = ['avg_price', 'std_dev']
    X = df_gpu[feature_cols]
    y_log = df_gpu['target']

    # 로그 값을 실제 가격으로 변환 (expm1)해서 실제 가격 기준으로 RMSE 계산
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, shuffle=False)
    y_test_real = np.expm1(y_test_log)

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        tree_method='hist',
        device='cpu',
        subsample=0.8,
    )

    model.fit(X_train.to_numpy(), y_train_log.to_numpy())
    y_pred_log = model.predict(X_test.to_numpy())
    y_pred_real = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    return model, df_gpu.iloc[-1:][feature_cols], rmse
