import pymysql
import os
from dbutils.pooled_db import PooledDB
dbhost = os.environ.get("DB_ADDR")
dbuser = os.environ.get('DB_USER')
dbpasswd = os.environ.get('DB_PASSWORD')
dbname = os.environ.get('DB_DBNANE')
conn = pymysql . connect(
    host=dbhost,  user=dbuser,  passwd=dbpasswd, charset="utf8")
cur = conn.cursor()
# cur.execute(f'DROP DATABASE {dbname}')
cur.execute(
    f"CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8 COLLATE utf8_general_ci")
cur.close()
conn.close()

pool = PooledDB(pymysql,
                host=dbhost, db=dbname,
                user=dbuser, passwd=dbpasswd,
                charset="utf8",
                ping=7,
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
                )
