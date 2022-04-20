import pymysql
import os
from dbutils.pooled_db import PooledDB

dbhost = os.environ.get("DB_ADDR")
dbuser = os.environ.get('DB_USER')
dbpasswd = os.environ.get('DB_PASSWORD')
dbname = os.environ.get('DB_DBNAME')
conn = pymysql . connect(
    host=dbhost,  user=dbuser,  passwd=dbpasswd, charset="utf8",
    ssl_ca=os.environ.get('SSL_CA'),
    ssl_key=os.environ.get('SSL_KEY'),
    ssl_cert=os.environ.get('SSL_CERT'))
cur = conn.cursor()
# cur.execute(f'DROP DATABASE {dbname}')
cur.execute(
    f"CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8 COLLATE utf8_general_ci")
# cur.execute("SHOW STATUS LIKE 'Ssl_cipher'") # check ssl connetion
# print(cur.fetchone())
cur.close()
conn.close()

pool = PooledDB(pymysql,
                host=dbhost, db=dbname,
                user=dbuser, passwd=dbpasswd,
                charset="utf8",
                ping=7,
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
                ssl_ca=os.environ.get('SSL_CA'),
                ssl_key=os.environ.get('SSL_KEY'),
                ssl_cert=os.environ.get('SSL_CERT')
                )
#   ssl = {
#         'ssl' : {
#             'ca' : '/home/yuan/server-ca.pem',
#             'key' : '/home/yuan/client-key.pem',
#             'cert' : '/home/yuan/client-cert.pem'
#         }
#     }
# conn = pool.connection()
# cur = conn.cursor()
# cur.execute("SHOW STATUS LIKE 'Ssl_cipher'")  # check ssl connetion
# check ssl connetion
# print(cur.fetchone())
