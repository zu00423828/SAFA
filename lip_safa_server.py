from animation_flow import make_image_animation_dataflow
from w2l.utils.face_detect import detect_face_and_dump_from_video
from w2l.utils.generate import generate_video
from datetime import datetime
from utils.mysql_dbtool import dbtools
import os
from io import BytesIO
import imageio

def generatefunction(data, GENERATE_BATCH_SIZE):
    video_name = ".".join(os.path.basename(data["video_path"]).split(".")[:-1])
    audio_name = ".".join(os.path.basename(data["audio_path"]).split(".")[:-1])
    filename = '{}_{}_{}_{}_{}.mp4'.format(
        video_name, audio_name, data["out_fps"], data["out_crf"], data["start_seconds"])

    destpath='datadir/dest/{}'.format(filename)
    dbtools.update_generate_job(
        data["id"],'lip start', start_datetime=datetime.now())
    generate_video(data["face_config_path"], data["audio_path"], "ckpt/wav2lip_gan.pth",
                   destpath, batch_size=GENERATE_BATCH_SIZE, face_fps=30, output_crf=data["out_crf"])
    dbtools.update_generate_job(
        data["id"], 'lip finish',filename=filename, dest_path=destpath, end_datetime=datetime.now())
    

def worker():
    FACE_DETECT_BATCH_SIZE=4
    GENERATE_BATCH_SIZE=32
    with dbtools.session() as sess:
        print(sess.processing_ticket_id)
        job=dbtools.get_job_join()
        print(job)
        if job is not None:
            tempdir = ".".join(os.path.basename(job["video_path"]).split(".")[:-1])
            dumpdir = os.path.join("datadir/preprocess", tempdir)
            face_config=os.path.join(dumpdir,'face.tsv')
            if not os.path.exists(face_config):
                face_config=detect_face_and_dump_from_video(job['video_path'],dumpdir,96, face_detect_batch_size=FACE_DETECT_BATCH_SIZE,smooth=True)
            generate_video(face_config,job['audio_path'],'/tmp/lip.mp4',model_path='ckpt',batchsize=GENERATE_BATCH_SIZE)
            result_path=''
            image_content=job['image_content']
            print(type(image_content))
            b=BytesIO(image_content)
            img=imageio.imread(b)
            make_image_animation_dataflow(img,'/tmp/lip.mp4',result_path,'ckpt',use_crop=True,use_gfp=job['enhance'])





if __name__ =='__main__':

    worker()
