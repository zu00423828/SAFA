from flask import Flask,request,render_template
import os
import json
from glob import glob
import subprocess
from animation_demo import main
app=Flask(__name__,static_url_path='/download',static_folder='finish/')


@app.route('/upload',methods=['POST'])
def upload_data():
    if not os.path.exists('upload'):
        os.mkdir('upload')
    source=request.files['source']
    driving=request.files['driving']
    source.save(os.path.join('upload',source.filename))
    driving.save(os.path.join('upload',driving.filename))
    return 'upload sussess',200
@app.route('/data')
def get_data():
    source_lists = glob('upload/*.png')
    source_lists.extend(glob('upload/*.jpg'))
    driving_lists = glob('upload/*.mp4')
    driving_lists.extend(glob('upload/*.mkv'))
    driving_lists.extend(glob('upload/*.avi'))
    print(source_lists,driving_lists)
    return json.dumps({'source':source_lists,'driving':driving_lists}),200
@app.route('/job',methods=['POST'])
def make_job():
    source_path=request.values['source']
    driving_path=request.values['driving']
    save_filename=request.values['save_filename']
    if not os.path.exists('finish'):
        os.mkdir('finish')
    save_filename=os.path.join('finish',save_filename)
    main(source_path,driving_path,save_filename,'config/end2end.yaml','ckpt/final_3DV.tar',with_eye=True,relative=True,adapt_scale=True)
    return  "sussess",200



@app.route('/')
def index():
    return render_template("upload.html")
@app.route('/select')
def select():
    return render_template('select.html')

if __name__=='__main__':
    app.run(debug=True)