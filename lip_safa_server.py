from animation_flow import make_image_animation_dataflow
from w2l.utils.face_detect import detect_face_and_dump_from_video
from w2l.utils.generate import generate_video
from datetime import datetime
from utils.mysql_dbtool import dbtools
import os

def preprocessfunction(data, FACE_DETECT_BATCH_SIZE):
    dbtools.update_generate_job_status(
        data['gj.id'], 'preprocess')
    dbtools.update_preprocess_job(
        data["id"], 'preprocessing', start_datetime=datetime.now())
    tempdir = ".".join(os.path.basename(data["path"]).split(".")[:-1])
    dumpdir = os.path.join("datadir/preprocess", tempdir)
    face_config_path = detect_face_and_dump_from_video(
        data["path"], dumpdir, "cuda", 96, face_detect_batch_size=FACE_DETECT_BATCH_SIZE)
    dbtools.update_preprocess_job(data["id"], 'dump',
                                  face_config_path=face_config_path, end_datetime=datetime.now())


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
        sess.processing_ticket_id
        preprocess_job=dbtools.get_preprocess_job_video()
        print(preprocess_job)
        if preprocess_job is not None:
            pass
            # preprocessfunction(preprocess_job,FACE_DETECT_BATCH_SIZE)
        generate_job=dbtools.get_job_join()
        print(generate_job)
        if generate_job is not None:
            pass
        # generatefunction(generate_job,GENERATE_BATCH_SIZE)
    # make_image_animation_dataflow(source_img,driving_video,result,model,use_crop,use_gfp)


# def animation_worker():

    
#         source_img=''
#         driving_video=''
#         result=''
#         model=''
#         use_crop=False
#         use_gfp=False
#         # lipserver_worker()
#         make_image_animation_dataflow(source_img,driving_video,result,model,use_crop,use_gfp)



if __name__ =='__main__':

    worker()
