from datetime import datetime, timedelta
from enum import Enum
import pymysql
import os
from uuid import uuid4

from .db import pool
# expiration_day=int(os.environ.get("EXPIRATION_DAY")) if os.environ.get("EXPIRATION_DAY") else 7
class PreprocessStatusEnum(Enum):
    init = "init"
    preprocessing = "preprocessing"
    dump = "dump"
    error = "error"


class GenerateStatusEnum(Enum):
    init = 'init'
    preprocess="preprocess"
    generate = 'generate'
    finish = 'finish'
    error = 'error'

class JobSession:
    def __init__(self,conn,cursor):
        self.conn=conn
        self.cursor=cursor
    def __enter__(self):

        value= uuid4().hex
        self.cursor.execute("INSERT  INTO processing_ticket   VALUES (%s,%s)",(None,value))

        self.cursor.execute("SELECT id FROM processing_ticket WHERE value=%s",(value))
        self.processing_ticket_id=self.cursor.fetchone()["id"]
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print("got exit error:")
            print("type:", exc_type)
            print("exc_val:", exc_val)
            print("exc_tb:", exc_tb)
            self.conn.rollback()

        try:
            
            self.cursor.execute("""DELETE FROM processing_ticket WHERE id = %(id)s""",
                        {'id': self.processing_ticket_id})
        except Exception as err:
            print("exit session failed: {}".format(err))

class DBtools:
    def __init__(self):
        connect=pool.connection()
        image='''CREATE TABLE IF NOT EXISTS image(
            id  INTEGER  PRIMARY KEY AUTO_INCREMENT,
            filename VARCHAR(200) UNIQUE NOT NULL,
            content  LONGBLOB   NOT NULL,
            md5 VARCHAR(32) UNIQUE NOT NULL,
            create_datetime  DATETIME NOT NULL,
            size_mb FLOAT NOT NULL,
            comment TEXT
            )'''
        video = '''CREATE TABLE IF NOT EXISTS video(
            id  INTEGER  PRIMARY KEY AUTO_INCREMENT,
            filename VARCHAR(200) UNIQUE NOT NULL,
            path  VARCHAR(200)  UNIQUE NOT NULL,
            md5 VARCHAR(32) UNIQUE NOT NULL,
            create_datetime  DATETIME NOT NULL,
            expiration_datetime  DATETIME NOT NULL,
            size_mb FLOAT NOT NULL,
            duration FLOAT NOT NULL,
            fps FLOAT NOT  NULL,
            comment TEXT
            )'''
        audio = '''CREATE TABLE IF NOT EXISTS audio (
            id  INTEGER PRIMARY KEY AUTO_INCREMENT,
            filename VARCHAR(200) UNIQUE NOT NULL,
            path  VARCHAR(200) UNIQUE  NOT NULL,
            md5 VARCHAR(32) UNIQUE NOT  NULL,
            create_datetime  DATETIME NOT NULL,
            expiration_datetime  DATETIME NOT NULL,
            size_mb  FLOAT NOT NULL,
            duration FLOAT NOT NULL,
            comment TEXT
            )'''

        generate_job = '''CREATE TABLE IF NOT EXISTS generate_job (
            id  INTEGER  PRIMARY KEY AUTO_INCREMENT,
            image_id INTEGER NOT NULL,
            video_id INTEGER NOT NULL,
            audio_id INTEGER NOT NULL,
            filename VARCHAR(200) UNIQUE ,
            face_config_path TEXT,
            path TEXT,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL,
            create_datetime DATETIME NOT NULL,
            start_datetime DATETIME ,
            end_datetime DATETIME,
            out_crf INTEGER NOT NULL,
            enhance BOOLEAN NOT NULL,
            comment TEXT NOT NULL,
            FOREIGN KEY(video_id) REFERENCES video(id) ON DELETE CASCADE,
            FOREIGN KEY(audio_id) REFERENCES audio(id) ON DELETE CASCADE,
            UNIQUE KEY unique_item(image_id,video_id,audio_id,out_crf,enhance)
            )'''
        tts_cache='''CREATE TABLE IF NOT EXISTS tts_cache(
            id INTEGER  PRIMARY KEY AUTO_INCREMENT,
            tts_content LONGBLOB NOT NULL,
            transform_text TEXT NOT NULL,
            lang VARCHAR(10) NOT NULL,
            voice VARCHAR(30) NOT NULL,
            rate FLOAT NOT NULL,
            create_datetime DATETIME NOT NULL,
            expiration_datetime DATETIME NOT NULL
            )'''
        processing_ticket='''CREATE TABLE IF NOT EXISTS processing_ticket(
            id INTEGER  PRIMARY KEY AUTO_INCREMENT,
            value CHAR(40) NOT NULL
        )'''
        cursor = connect.cursor()
        cursor.execute(image)
        cursor.execute(video)
        cursor.execute(audio)
        cursor.execute(generate_job)
        cursor.execute(tts_cache)
        cursor.execute(processing_ticket)
        self.close(connect,cursor)
    def create_conn_cursor(self):
        connect=pool.connection()
        cursor=connect.cursor()
        return connect,cursor
    def close(self,connect,cursor):
        cursor.close()
        connect.close()

    def get_generate_job(self,id):
        connect,cursor=self.create_conn_cursor()
        # select_query = '''SELECT gj.id,video.path as video_path,audio.path as audio_path,gj.path as  dest_path,gj.out_fps,gj.out_crf,start_seconds,gj.status,\
        #     gj.create_datetime,gj.end_datetime,gj.comment FROM generate_job as gj INNER JOIN video ON gj.video_id=video.id \
        #     INNER JOIN audio ON gj.audio_id=audio.id INNER JOIN preprocess_job as pj \
        #     ON pj.video_id=video.id ORDER BY gj.id ASC'''
        select_query=f'SELECT gj.status as gj_status,pj.status as pj_status  FROM generate_job as gj INNER JOIN preprocess_job as pj ON gj.video_id=pj.video_id and gj.id={id}'
        cursor.execute(select_query)
        result = cursor.fetchone()
        self.close(connect,cursor)
        return result

    def get_lip_job_join(self):
        connect,cursor=self.create_conn_cursor()
        select_query="SELECT gj.id,video.path video_path ,audio.path audio_path,video.id ,audio.id,pj.face_config_path,gj.out_fps,gj.out_crf,gj.start_seconds FROM generate_job as gj \
            INNER JOIN video ON gj.video_id=video.id \
            INNER JOIN audio ON gj.audio_id=audio.id INNER JOIN preprocess_job as pj \
            ON pj.video_id=video.id  WHERE (gj.status='init' OR gj.status='preprocess')AND pj.status='dump' ORDER BY gj.id ASC"
        cursor.execute(select_query)
        result=cursor.fetchone()
        self.close(connect,cursor)
        return result
    def get_job_join(self):
        connect,cursor=self.create_conn_cursor()
        select_query="SELECT gj.id,video.path video_path ,audio.path audio_path,video.id ,audio.id,gj.out_crf,gj.enhance,image.content image_content FROM generate_job as gj \
            INNER JOIN  image ON gj.image_id =image.id\
            INNER JOIN video ON gj.video_id=video.id \
            INNER JOIN audio ON gj.audio_id=audio.id \
            WHERE gj.status!='finish' ORDER BY gj.id ASC"
        cursor.execute(select_query)
        result=cursor.fetchone()
        self.close(connect,cursor)
        return result

    def session(self):
        conn,cursor=self.create_conn_cursor()
        return JobSession(conn,cursor)
dbtools = DBtools()
