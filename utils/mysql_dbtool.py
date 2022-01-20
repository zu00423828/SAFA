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
        preprocess_job = '''CREATE TABLE  IF NOT EXISTS preprocess_job (
            id  INTEGER  PRIMARY KEY AUTO_INCREMENT,
            video_id  INTEGER UNIQUE NOT NULL,
            face_config_path TEXT,
            status  TEXT NOT NULL,
            create_datetime DATETIME NOT NULL,
            start_datetime DATETIME ,
            end_datetime DATETIME ,
            FOREIGN KEY(video_id) REFERENCES video(id) ON DELETE CASCADE
            )'''
            # ,
            # FOREIGN KEY(video_id) REFERENCES video(id)
        generate_job = '''CREATE TABLE IF NOT EXISTS generate_job (
            id  INTEGER  PRIMARY KEY AUTO_INCREMENT,
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
        cursor.execute(preprocess_job)
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
    # 共用
    def get_table_info(self, table_name, id=None,reverse=False):
        connect,cursor=connect,cursor=self.create_conn_cursor()
        if id is None and reverse==False:
            cursor.execute(f"SELECT * FROM {table_name}")
        elif reverse:
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC")
        else:
            cursor.execute(f"SELECT * FROM {table_name} WHERE id={id}")
        result = cursor.fetchall() if id is  None else cursor.fetchone()
        self.close(connect,cursor)
        return result

    def get_video_audio_md5(self, table_name, md5):
        connect,cursor=connect,cursor=self.create_conn_cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE md5='{md5}'")
        result = cursor.fetchone()
        self.close(connect,cursor)
        return False if not result  else True
        #return True if len(result)==0 else False
    def get_expiration(self, table_name, id=None):
        connect,cursor=connect,cursor=self.create_conn_cursor()
        cursor.execute(
            f"SELECT id,expiration_datetime FROM {table_name}")
        result = cursor.fetchall()
        self.close(connect,cursor)
        return result

    def update_video_audio_expiration_datetime(self, table_name,id, expiration_datetime):
        connect,cursor=connect,cursor=self.create_conn_cursor()
        update_query = f"UPDATE {table_name} SET expiration_datetime='{expiration_datetime}' WHERE id={id}"
        cursor.execute(update_query)
        self.close(connect,cursor)
        # self.conn.commit()

    def update_comment(self, table_name, id, comment):
        connect,cursor=self.create_conn_cursor()
        update_query = f"UPDATE {table_name} SET comment='{comment}' WHERE id={id}"
        cursor.execute(update_query)
        self.close(connect,cursor)

    def delete_data(self, table_name, id):
        connect,cursor=self.create_conn_cursor()
        select_query = f"SELECT * from {table_name} WHERE id={id}"
        cursor.execute(select_query)
        result = cursor.fetchone()
        cursor.execute(f"DELETE FROM {table_name} WHERE id={id}")
        self.close(connect,cursor)
        return result
    # 不共用
    def add_video(self, data):
        connect,cursor=self.create_conn_cursor()
        insert_query = "INSERT INTO video VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor.execute(insert_query, data)
            print(data[1])
            cursor.execute(f"SELECT * FROM video WHERE filename='{data[1]}'")
        except Exception as err:
            print(err)
        # self.conn.commit()
        result=cursor.fetchone()
        # self.pool.release(self.conn)
        self.close(connect,cursor)
        return result

    def add_audio(self, data):
        connect,cursor=self.create_conn_cursor()
        insert_query = "INSERT INTO audio VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor.execute(insert_query, data)
            cursor.execute(f"SELECT * FROM audio WHERE filename='{data[1]}'")
        except Exception as err:
            print(err)
        result=cursor.fetchone()
        self.close(connect,cursor)
        return result 

    def add_preprocess_job(self, video_id):
        connect,cursor=self.create_conn_cursor()
        insert_query = "INSERT INTO preprocess_job VALUES(%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor.execute(insert_query,(None, video_id,None,
                                             PreprocessStatusEnum.init.value, datetime.now(), None, None))
        except Exception as err:
            print("preprocess",err)
            cursor.execute(f"SELECT * FROM preprocess_job WHERE video_id={video_id}")
            result=cursor.fetchone()
            self.close(connect,cursor)
            return 0,result
        cursor.execute(f"SELECT * FROM preprocess_job WHERE video_id={video_id}")
        result=cursor.fetchone()
        self.close(connect,cursor)
        return 1,result
    def add_generate_job(self, video_id,audio_id,out_fps=None,out_crf=0,start_seconds=0,comment=" "):
        connect,cursor=self.create_conn_cursor()
        insert_query="INSERT INTO generate_job VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor.execute(insert_query,(None, video_id,
                                                 audio_id, None,None, GenerateStatusEnum.init.value,0,datetime.now(), None, None, out_fps, out_crf, start_seconds,comment))
        except Exception as err:
            print("generate",err)
            self.close(connect,cursor)
            return None
        cursor.execute("SELECT * FROM generate_job ORDER BY id desc ")
        result=cursor.fetchone()
        self.close(connect,cursor)
        return result 
    def add_tts_cache(self,tts_content,transform_text,lang,voice,rate):
        connect,cursor=self.create_conn_cursor()
        insert_query="INSERT INTO tts_cache VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor.execute(insert_query,(None,tts_content,transform_text,lang,voice,rate,datetime.now(),datetime.now()+timedelta(days=expiration_day)))
        except Exception as err:
            print(err)
        self.close(connect,cursor)    

    def get_tts(self,transform_text,lang,voice,rate):
        connect,cursor=self.create_conn_cursor()
        select_query="SELECT * FROM tts_cache WHERE transform_text= %s AND lang=%s AND voice=%s AND rate LIKE %s"
        cursor.execute(select_query,(transform_text,lang,voice,rate))
        result=cursor.fetchone()
        self.close(connect,cursor)
        return result
    def update_preprocess_job(self, id, status, face_config_path=None, start_datetime=None, end_datetime=None):
        connect,cursor=self.create_conn_cursor()
        if start_datetime is not None:
            update_query = f"UPDATE preprocess_job SET status='{status}',start_datetime='{start_datetime}' WHERE id={id}"
        else:
            update_query = f"UPDATE preprocess_job SET status='{status}',face_config_path='{face_config_path}',end_datetime= '{end_datetime}' WHERE id={id}"
        cursor.execute(update_query)
        self.close(connect,cursor)
    def update_jobprogress(self,id,progress):
        connect,cursor=self.create_conn_cursor()
        update_query=f"UPDATE generate_job SET progress={progress} WHERE id={id}"
        cursor.execute(update_query)
        self.close(connect,cursor)

    def update_generate_job_status(self,id,status):
        connect,cursor=self.create_conn_cursor()
        cursor.execute(f"UPDATE generate_job SET status='{status}' WHERE id={id}")
        self.close(connect,cursor)
    def update_generate_job(self, id, status, filename=None,dest_path=None, start_datetime=None, end_datetime=None):
        connect,cursor=self.create_conn_cursor()        
        if start_datetime is not None:
            update_query = f"UPDATE generate_job SET status='{status}',start_datetime= '{start_datetime}' WHERE id={id}"
        else:
            update_query = f"UPDATE generate_job SET filename='{filename}',path='{dest_path}', status='{status}',end_datetime= '{end_datetime}' WHERE id={id}"
        cursor.execute(update_query)
        self.close(connect,cursor)

    def get_preprocess_job_video(self):
        connect,cursor=self.create_conn_cursor()        
        select_query = "SELECT pj.id,video.path,gj.id  FROM preprocess_job as pj INNER JOIN video,generate_job as gj WHERE pj.video_id=video.id AND gj.video_id=video.id AND pj.status!='dump'"
        cursor.execute(select_query)
        result = cursor.fetchone()
        self.close(connect,cursor)
        return result

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
        select_query="SELECT gj.id,video.path video_path ,audio.path audio_path,video.id ,audio.id,pj.face_config_path,gj.out_crf,gj.enhance,image.content imagee_content FROM generate_job as gj \
            INNER JOIN  image ON gj.image_id =image.id\
            INNER JOIN video ON gj.video_id=video.id \
            INNER JOIN audio ON gj.audio_id=audio.id \
            INNER JOIN preprocess_job as pj \
            ON pj.video_id=video.id  WHERE (gj.status='init' OR gj.status='preprocess')AND pj.status='dump' ORDER BY gj.id ASC"
        cursor.execute(select_query)
        result=cursor.fetchone()
        self.close(connect,cursor)
        return result
    def get_preprocess_dump_generate_finish(self):
        connect,cursor=self.create_conn_cursor()
        select_query_preprocess="SELECT video.id,video.size_mb,video.duration,TIMESTAMPDIFF(SECOND,start_datetime,end_datetime) as job_duration from preprocess_job as pj INNER JOIN video ON pj.video_id=video.id  AND pj.status='dump'"
        cursor.execute(select_query_preprocess)
        preprocess_dump=cursor.fetchall()
        select_query_generate="SELECT audio.id,audio.size_mb,audio.duration,TIMESTAMPDIFF(SECOND,start_datetime,end_datetime) as job_duration from generate_job as gj INNER JOIN audio ON gj.audio_id=audio.id  AND gj.status='finish'"
        cursor.execute(select_query_generate)
        generate_finish=cursor.fetchall()
        self.close(connect,cursor)
        return preprocess_dump,generate_finish
    def get_preprocess_not_dump_generate_not_finish(self,preprocess_key,preprocess_id):
        connect,cursor=self.create_conn_cursor()        
        if preprocess_key:
            select_query_preprocess=f"SELECT video.id,video.size_mb,video.duration,TIMESTAMPDIFF(SECOND,start_datetime,end_datetime) as job_duration from preprocess_job as pj INNER JOIN video ON pj.video_id=video.id  AND pj.status!='dump' and pj.id!={preprocess_id}"
        else:
            select_query_preprocess="SELECT video.id,video.size_mb,video.duration,TIMESTAMPDIFF(SECOND,start_datetime,end_datetime) as job_duration from preprocess_job as pj INNER JOIN video ON pj.video_id=video.id  AND pj.status!='dump'"
        cursor.execute(select_query_preprocess)
        preprocess_not_dump=cursor.fetchall()
        select_query_generate="SELECT audio.id,audio.size_mb,audio.duration,TIMESTAMPDIFF(SECOND,start_datetime,end_datetime) as job_duration from generate_job as gj INNER JOIN audio ON gj.audio_id=audio.id  AND gj.status!='finish'"
        cursor.execute(select_query_generate)
        generate_not_finish=cursor.fetchall()
        self.close(connect,cursor)
        return preprocess_not_dump,generate_not_finish
    def session(self):
        conn,cursor=self.create_conn_cursor()
        return JobSession(conn,cursor)
dbtools = DBtools()
