import pymysql
from sqlalchemy import create_engine

# DB 설정을 한 곳에서 관리
DB_USER = 'root'
DB_PASSWORD = '0000'
DB_NAME = 'test2'
DB_HOST = 'localhost'
DB_CHARSET = 'utf8'

def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset=DB_CHARSET
    )