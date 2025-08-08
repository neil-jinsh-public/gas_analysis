# database_model.py

import datetime
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dateutil.relativedelta import relativedelta
from datetime import timedelta

def db_conn():
    """
    连接 PostgreSQL 数据库。
    """
    try:
        # engine = create_engine("postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/ykrq")
        password = quote_plus("zjplan@2021")  # 自动编码
        engine = create_engine(f"mysql+pymysql://root:{password}@localhost:3306/zj_csaq_3303?charset=utf8mb4")
        return engine.connect()  # 返回一个连接对象
    except Exception as e:
        print("Error connecting to database:", e)
        return None

def get_gas_cylinders(conn):
    """
    分批读取气瓶库表，带进度信息显示
    """
    print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始从数据库读取气瓶数据...")

    chunks = []
    batch_size = 500000
    offset = 0
    total_rows = 0
    pbar = tqdm(desc="正在读取气瓶数据", unit="条", dynamic_ncols=True)

    while True:
        query = f"SELECT * FROM zj_csaq_3303.pzrq_gas_cylinder_temp LIMIT {batch_size} OFFSET {offset}"
        # query = f"SELECT * FROM zj_csaq_3303.pzrq_gas_cylinder_temp LIMIT {batch_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query, conn)

        if df_chunk.empty:
            break

        chunks.append(df_chunk)
        offset += batch_size
        total_rows += len(df_chunk)
        pbar.update(len(df_chunk))  # 手动更新进度条

    pbar.close()

    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取数据{len(df_full)}条...")
        return df_full
    else:
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取数据0条...")
        return pd.DataFrame()

def get_gas_status(conn):
    """
    分批读取气瓶库表，带进度信息显示
    """
    print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始从数据库读取气瓶流转数据...")

    chunks = []
    batch_size = 500000
    offset = 0
    total_rows = 0
    pbar = tqdm(desc="正在读取气瓶流转数据", unit="条", dynamic_ncols=True)

    while True:
        query = f"SELECT * FROM zj_csaq_all.t_gas_status LIMIT {batch_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query, conn)

        if df_chunk.empty:
            break

        chunks.append(df_chunk)
        offset += batch_size
        total_rows += len(df_chunk)
        pbar.update(len(df_chunk))  # 手动更新进度条

    pbar.close()

    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取流转数据{len(df_full)}条...")
        return df_full
    else:
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取流转数据0条...")
        return pd.DataFrame()

def get_gas_scanning(conn, start_time):
    """
    分批读取气瓶库表，带进度信息显示
    """
    print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始从数据库读取气瓶流转扫描数据...")

    chunks = []
    batch_size = 500000
    offset = 0
    total_rows = 0
    pbar = tqdm(desc="正在读取气瓶流转数据", unit="条", dynamic_ncols=True)

    while True:
        query = f"SELECT * FROM zj_csaq_3303.t_gas_bussiness_log_temp LIMIT {batch_size} OFFSET {offset}"
        # query = f"SELECT * FROM zj_csaq_3303.t_gas_bussiness_log WHERE createtime between '{start_time}' and '{start_time + relativedelta(months=1)}' LIMIT {batch_size} OFFSET {offset}"
        # query = f"SELECT * FROM zj_csaq_3303.t_gas_bussiness_log_temp LIMIT {batch_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query, conn)

        if df_chunk.empty:
            break

        chunks.append(df_chunk)
        offset += batch_size
        total_rows += len(df_chunk)
        pbar.update(len(df_chunk))  # 手动更新进度条

    pbar.close()

    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取流转扫描数据{len(df_full)}条...")
        return df_full
    else:
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取流转扫描数据0条...")
        return pd.DataFrame()

def get_employee(conn):
    """
        分批读取员工库表，带进度信息显示
    """
    print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始从数据库读取员工数据...")

    chunks = []
    batch_size = 500000
    offset = 0
    total_rows = 0
    pbar = tqdm(desc="正在读取员工数据", unit="条", dynamic_ncols=True)

    while True:
        query = f"SELECT * FROM zj_csaq_3303.pzrq_gas_employee where cscode='330300' LIMIT {batch_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query, conn)

        if df_chunk.empty:
            break

        chunks.append(df_chunk)
        offset += batch_size
        total_rows += len(df_chunk)
        pbar.update(len(df_chunk))  # 手动更新进度条

    pbar.close()

    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取员工数据{len(df_full)}条...")
        return df_full
    else:
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取员工数据0条...")
        return pd.DataFrame()

def get_gas_company(conn):
    """
            分批读取企业库表
        """
    print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始从数据库读取企业数据...")

    chunks = []
    batch_size = 500000
    offset = 0
    total_rows = 0
    pbar = tqdm(desc="正在读取员工数据", unit="条", dynamic_ncols=True)

    while True:
        query = f"SELECT * FROM zj_csaq_3303.pzrq_gas_company where cscode='330300' LIMIT {batch_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query, conn)

        if df_chunk.empty:
            break

        chunks.append(df_chunk)
        offset += batch_size
        total_rows += len(df_chunk)
        pbar.update(len(df_chunk))  # 手动更新进度条

    pbar.close()

    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取企业数据{len(df_full)}条...")
        return df_full
    else:
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取企业数据0条...")
        return pd.DataFrame()


def get_customer(conn):
    """
                分批读取企业库表
            """
    print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始从数据库读取企业数据...")

    chunks = []
    batch_size = 500000
    offset = 0
    total_rows = 0
    pbar = tqdm(desc="正在读取客户数据", unit="条", dynamic_ncols=True)

    while True:
        query = f"SELECT * FROM zj_csaq_3303.pzrq_gas_customer_temp where cscode='330300' LIMIT {batch_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query, conn)

        if df_chunk.empty:
            break

        chunks.append(df_chunk)
        offset += batch_size
        total_rows += len(df_chunk)
        pbar.update(len(df_chunk))  # 手动更新进度条

    pbar.close()

    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取客户数据{len(df_full)}条...")
        return df_full
    else:
        print(f">>> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 共读取客户数据0条...")
        return pd.DataFrame()
