from datetime import datetime, timedelta
from enum import Enum
import pymysql
import os
from uuid import uuid4

from .db import pool
# expiration_day=int(os.environ.get("EXPIRATION_DAY")) if os.environ.get("EXPIRATION_DAY") else 7

class JobSession:
    def __init__(self):
        pass

    def __enter__(self):
        connect = pool.connection()
        cursor = connect.cursor()
        value = uuid4().hex
        cursor.execute(
            "INSERT  INTO processing_ticket   VALUES (%s,%s,%s)", (None, value, None))

        cursor.execute(
            "SELECT id FROM processing_ticket WHERE value=%s", (value))
        self.processing_ticket_id = cursor.fetchone()["id"]
        cursor.close()
        connect.close()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if exc_type is not None:
        #     print("got exit error:")
        #     print("type:", exc_type)
        #     print("exc_val:", exc_val)
        #     print("exc_tb:", exc_tb)
        #     self.conn.rollback()

        try:
            connect = pool.connection()
            cursor = connect.cursor()
            cursor.execute("""DELETE FROM processing_ticket WHERE id = %(id)s""",
                           {'id': self.processing_ticket_id})
            cursor.close()
            connect.close()
        except Exception as err:
            print("exit session failed: {}".format(err))


class DBtools:
    def __init__(self):
        connect = pool.connection()
        client = '''
            CREATE TABLE IF NOT EXISTS `client` (
            `id` INTEGER PRIMARY KEY AUTO_INCREMENT,
            `account` CHAR(128) NOT NULL,
            `salt` CHAR(32) NOT NULL,
            `pwd` CHAR(64) NOT NULL,
            `create_datetime` DATETIME NOT NULL,
            `comment` TEXT NULL,
            UNIQUE KEY `uniq_salt` (salt(32)),
            UNIQUE KEY `uniq_account` (account(128))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''
        image = '''CREATE TABLE IF NOT EXISTS image (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            `client_id` INTEGER NOT NULL,
            filename VARCHAR(200) NOT NULL,
            display_image_content LONGBLOB NOT NULL,
            generate_image_content LONGBLOB NOT NULL,
            md5 VARCHAR(32) NOT NULL,
            create_datetime DATETIME NOT NULL,
            size_mb FLOAT NOT NULL,
            comment TEXT,
            FOREIGN KEY(client_id) REFERENCES client(id) ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE KEY `uniq_client_md5` (client_id, md5(32))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''
        video = '''CREATE TABLE IF NOT EXISTS video (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            `client_id` INTEGER NOT NULL,
            filename VARCHAR(200) NOT NULL,
            path VARCHAR(200) UNIQUE NOT NULL,
            md5 VARCHAR(32) NOT NULL,
            create_datetime  DATETIME NOT NULL,
            expiration_datetime  DATETIME ,
            size_mb FLOAT NOT NULL,
            duration FLOAT NOT NULL,
            fps FLOAT NOT NULL,
            comment TEXT,
            FOREIGN KEY(client_id) REFERENCES client(id) ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE KEY `uniq_client_md5` (client_id, md5(32))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''
        audio = '''CREATE TABLE IF NOT EXISTS audio (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            `client_id` INTEGER NOT NULL,
            filename VARCHAR(200) NOT NULL,
            path VARCHAR(200) UNIQUE NOT NULL,
            md5 VARCHAR(32) NOT NULL,
            create_datetime DATETIME NOT NULL,
            expiration_datetime  DATETIME ,
            size_mb FLOAT NOT NULL,
            duration FLOAT NOT NULL,
            comment TEXT,
            FOREIGN KEY(client_id) REFERENCES client(id) ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE KEY `uniq_client_md5` (client_id, md5(32))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''

        generate_job = '''CREATE TABLE IF NOT EXISTS generate_job (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            image_id INTEGER NOT NULL,
            video_id INTEGER NOT NULL,
            audio_id INTEGER NOT NULL,
            filename VARCHAR(200) UNIQUE ,
            path TEXT,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL,
            create_datetime DATETIME NOT NULL,
            start_datetime DATETIME ,
            end_datetime DATETIME,
            out_crf INTEGER NOT NULL,
            enhance BOOLEAN NOT NULL,
            comment TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES image(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY(video_id) REFERENCES video(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY(audio_id) REFERENCES audio(id) ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE KEY unique_item (image_id, video_id, audio_id, out_crf, enhance)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''
        tts_cache = '''CREATE TABLE IF NOT EXISTS tts_cache(
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            tts_content LONGBLOB NOT NULL,
            transform_text TEXT NOT NULL,
            lang VARCHAR(10) NOT NULL,
            voice VARCHAR(30) NOT NULL,
            rate FLOAT NOT NULL,
            create_datetime DATETIME NOT NULL,
            expiration_datetime DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''
        processing_ticket = '''CREATE TABLE IF NOT EXISTS processing_ticket(
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            value CHAR(40) NOT NULL,
            generate_job_id INTEGER,
            FOREIGN KEY(generate_job_id) REFERENCES generate_job(id) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4'''

        auth = '''
            CREATE TABLE IF NOT EXISTS `auth` (
            `id` INTEGER PRIMARY KEY AUTO_INCREMENT,
            `client_id` INTEGER NOT NULL,
            `access_token` CHAR(32) NOT NULL,
            `refresh_token` CHAR(32) NOT NULL,
            `count` INT NOT NULL DEFAULT 0,
            `activate` BOOLEAN NOT NULL DEFAULT true,
            `expiration_datetime` DATETIME NOT NULL,
            `create_datetime` DATETIME NOT NULL,
            UNIQUE KEY `uniq_access_token` (access_token(32)),
            UNIQUE KEY `uniq_refresh_token` (refresh_token(32)),
            FOREIGN KEY(client_id) REFERENCES client(id) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        '''
        cursor = connect.cursor()
        cursor.execute(client)
        cursor.execute(image)
        cursor.execute(video)
        cursor.execute(audio)
        cursor.execute(generate_job)
        cursor.execute(tts_cache)
        cursor.execute(processing_ticket)
        cursor.execute(auth)
        self.close(connect, cursor)

    def create_conn_cursor(self):
        connect = pool.connection()
        cursor = connect.cursor()
        return connect, cursor

    def close(self, connect, cursor):
        cursor.close()
        connect.close()

    def clear_data(self):
        connect, cursor = self.create_conn_cursor()
        cursor.execute('TRUNCATE TABLE processing_ticket')
        cursor.execute('TRUNCATE TABLE generate_job')
        cursor.execute('TRUNCATE TABLE image')
        cursor.execute('TRUNCATE TABLE video')
        cursor.execute('TRUNCATE TABLE audio')
        self.close(connect, cursor)

    def insert_client(self, data):
        connect, cursor = self.create_conn_cursor()
        insert_query = 'INSERT INTO client VALUES(%s,%s,%s,%s,%s,%s)'
        cursor.execute(insert_query, data)
        self.close(connect, cursor)

    def insert_image(self, data):
        connect, cursor = self.create_conn_cursor()
        insert_query = 'INSERT INTO image VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        try:
            cursor.execute(insert_query, data)
        except Exception as e:
            print(e)
        self.close(connect, cursor)

    def insert_video(self, data):
        connect, cursor = self.create_conn_cursor()
        insert_query = 'INSERT INTO video VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        try:
            cursor.execute(insert_query, data)
        except Exception as e:
            print(e)
        self.close(connect, cursor)

    def insert_audio(self, data):
        connect, cursor = self.create_conn_cursor()
        insert_query = 'INSERT INTO audio VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        try:
            cursor.execute(insert_query, data)
        except Exception as e:
            print(e)
        self.close(connect, cursor)

    def insert_job(self, data):
        connect, cursor = self.create_conn_cursor()
        insert_query = 'INSERT INTO generate_job VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        try:
            cursor.execute(insert_query, data)
        except Exception as e:
            print(e)
        self.close(connect, cursor)

    def get_data(self, table_name, where_args=None, all=True):
        connect, cursor = self.create_conn_cursor()
        if where_args == None:
            select_query = f'SELECT * FROM {table_name} '
        else:
            select_query = f'SELECT * FROM {table_name} WHERE {where_args} '
        cursor.execute(select_query)
        if all:
            result = cursor.fetchall()
        else:
            result = cursor.fetchone()
        self.close(connect, cursor)
        return result

    def get_job_join(self):
        connect, cursor = self.create_conn_cursor()
        select_query = "SELECT gj.id,video.path video_path ,audio.path audio_path,video.id ,audio.id,gj.out_crf,\
            gj.enhance,image.generate_image_content image_content,image.filename image_filename,\
            video.filename video_filename,audio.filename audio_filename\
            FROM generate_job as gj \
            INNER JOIN  image ON gj.image_id =image.id\
            INNER JOIN video ON gj.video_id=video.id \
            INNER JOIN audio ON gj.audio_id=audio.id \
            WHERE gj.status!='finish' ORDER BY gj.id ASC"
        cursor.execute(select_query)
        result = cursor.fetchone()
        self.close(connect, cursor)
        return result

    def update_job_progress(self, id, status, progress):
        connect, cursor = self.create_conn_cursor()
        update_query = 'UPDATE generate_job SET progress=%s,status=%s WHERE id=%s'
        cursor.execute(update_query, (progress, status, id))
        self.close(connect, cursor)

    def update_job_process_datetime(self, id, start):
        connect, cursor = self.create_conn_cursor()
        if start:
            update_query = 'UPDATE generate_job SET start_datetime=%s WHERE id=%s'
        else:
            update_query = 'UPDATE generate_job SET end_datetime=%s WHERE id=%s'
        cursor.execute(update_query, (datetime.now(), id))
        self.close(connect, cursor)

    def update_job_result(self, id, filename, path):
        connect, cursor = self.create_conn_cursor()
        update_query = 'UPDATE generate_job SET filename=%s,path=%s WHERE id=%s'
        cursor.execute(update_query, (filename, path, id))
        self.close(connect, cursor)

    def session(self):
        # conn,cursor=self.create_conn_cursor()
        return JobSession()

    def set_ticket_job(self, ticket_id, job_id):
        connect, cursor = self.create_conn_cursor()
        select_query = f'SELECT * FROM processing_ticket  WHERE generate_job_id={job_id}'
        cursor.execute(select_query)
        result=cursor.fetchone()
        if result is None:
            update_query = f"UPDATE processing_ticket SET generate_job_id={job_id} WHERE id={ticket_id}"
            cursor.execute(update_query)
            self.close(connect, cursor)
            return True
        else:
            self.close(connect, cursor)
            return False


dbtools = DBtools()
