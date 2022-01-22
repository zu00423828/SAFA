import pymysql
import os
from dbutils.pooled_db import PooledDB
if os.environ.get("DB_PORT_3306_TCP_ADDR") is None:
    dbhost = '34.80.95.212'
else:
    dbhost = os.environ.get("DB_PORT_3306_TCP_ADDR")
if os.environ.get('DB_ENV_MYSQL_ROOT_PASSWORD') is None:
    dbuser = 'lip'
else:
    dbuser = os.environ.get('DB_ENV_MYSQL_USER')

if os.environ.get('DB_ENV_MYSQL_ROOT_PASSWORD') is None:
    dbpasswd = 'HJg5qVASwKy3PVVH'
else:
    dbpasswd = os.environ.get('DB_ENV_MYSQL_ROOT_PASSWORD')
dbname = "test_image_lip_animation"
conn = pymysql . connect(
    host=dbhost,  user=dbuser,  passwd=dbpasswd, charset="utf8")
cur = conn.cursor()
cur.execute(
    f"DROP DATABASE  {dbname}")
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
