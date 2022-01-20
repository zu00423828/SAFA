import pymysql
import os
from dbutils.pooled_db import PooledDB
if os.environ.get("DB_PORT_3306_TCP_ADDR") is None:
    dbhost='127.0.0.1'
else:
    dbhost=os.environ.get("DB_PORT_3306_TCP_ADDR")
if os.environ.get('DB_ENV_MYSQL_ROOT_PASSWORD') is None:
    dbpasswd='zu7957232'
else:
    dbpasswd=os.environ.get('DB_ENV_MYSQL_ROOT_PASSWORD')
dbname = "animation_data"
conn = pymysql . connect(
            host=dbhost,  user='root',  passwd=dbpasswd, charset="utf8")
cur = conn.cursor()
cur.execute(
    f"CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8 COLLATE utf8_general_ci")
cur.close()
conn.close()

pool = PooledDB(pymysql,
                        host=dbhost, db=dbname,
                        user="root", passwd=dbpasswd,
                        charset="utf8",
                        ping=7,
                        cursorclass=pymysql.cursors.DictCursor,
                        autocommit=True
                        )

