from flaskr import app, model
from flask import Flask, request, redirect, url_for, render_template, Response
import zipfile
import plotly
import json
import os
import numpy as np

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

OUTPUT_DIR = 'tmp'

import shutil

def empty_folder(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = uploaded_file.filename
        if filename != '':
            if uploaded_file.filename[-3:] in ['zip']:
                image_path = os.path.join(OUTPUT_DIR, filename)
                uploaded_file.save(image_path)

                with zipfile.ZipFile(image_path, 'r') as zip_ref:
                    zip_ref.extractall(OUTPUT_DIR)

                img_dir = OUTPUT_DIR + "/" + filename[:-4]
                dicom_paths = model.get_dicom_paths(img_dir)

                pet_scan = model.get_pet_scan(dicom_paths)
                heat_map = model.compute_heatmap(pet_scan)

                prediction = model.get_prediction(pet_scan)
                if prediction.squeeze() > 0.5:
                    diagnosis = 'AD'
                else:
                    diagnosis = 'CN'

                fig = model.create_plot(pet_scan, heat_map)
                redata = json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder))
                relayout = json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder))

                fig_json=json.dumps({'data': redata,'layout': relayout})

                empty_folder(OUTPUT_DIR)

                return render_template('show.html', prediction = prediction, diagnosis = diagnosis, plot_json = fig_json)
    return render_template('index.html')
