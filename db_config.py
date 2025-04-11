import pymysql
from sqlalchemy import create_engine

def get_engine():
    return create_engine("mysql+pymysql://root:0000@localhost/danawa_crawler_data")

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='0000',
        db='danawa_crawler_data',
        charset='utf8'
    )