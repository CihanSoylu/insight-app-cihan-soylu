import tensorflow as tf
from PIL import Image
import io
import numpy as np
import pandas as pd

import os
import string
import random
import json
import requests
import glob
import pydicom
from scipy import ndimage, misc

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

model = tf.keras.models.load_model('flaskr/static/model/3dcnn_model_gap_98_78_min_loss.h5')

def get_dicom_paths(img_dir):
    dicom_paths = glob.glob(img_dir+'/*')

    slices = []
    for path in dicom_paths:
        n_slice = path.split('_')[-3]
        left = path.find('_' + n_slice + '_') + 1
        right = left + len(n_slice)
        slices.append([path[:left], int(n_slice), path[right:]])

    # Order slices
    slice_df = pd.DataFrame(slices, columns = ['col1', 'slice', 'col3']).sort_values(by='slice')

    slice_df['combined'] = slice_df['col1'] + slice_df['slice'].map(lambda x: str(x)) + slice_df['col3']

    return slice_df['combined'].values

def get_pet_scan(dicom_paths):
    slices = []
    for path in dicom_paths:
        img = pydicom.read_file(path)
        img_array = img.pixel_array

        img_array = np.expand_dims(img_array, axis = -1)
        img_array = tf.cast(img_array, tf.float32)
        slices.append(img_array)

    # Take 75 slices by pruning the first 11 and last 10 slices.
    pet_scan = np.concatenate(slices[11:86], axis=-1)
    #pet_scan[pet_scan < 500] = 0

    # Scale the images.
    minimum = np.min(pet_scan)
    maximum = np.max(pet_scan)
    pet_scan = (pet_scan - minimum)/(maximum - minimum)

    # Crop out the empty part of the image
    pet_scan = pet_scan[30:128,40:118,:]

    return pet_scan

def get_prediction(pet_scan):
    pet_scan = np.expand_dims(pet_scan, axis=0)

    prediction = model.predict(pet_scan)
    return prediction


def create_plot(petscan, heatmap, num_slices = 75):

    fig = make_subplots(rows=1, cols=2)
    # Create figure
    #fig = go.Figure()
    zmax = np.max(heatmap)
    zmin = np.min(heatmap)

    # Add traces, one for each slider step
    for step in range(num_slices):
        fig.add_trace(
            go.Heatmap(
                z=petscan[::-1,:,step], colorscale = 'Rainbow', showscale = False),
            row=1, col=1)
    for step in range(num_slices):
        fig.add_trace(
            go.Heatmap(
                z=heatmap[::-1,:,step], colorscale = 'viridis', zmax = zmax, zmin = zmin),
            row=1, col=2)

    fig.data[11].visible = True
    fig.data[11 + num_slices].visible = True


    # Create and add slider
    steps = []
    for i in range(num_slices):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to slice: " + str(i)}],  # layout attribute
        )
        step['args'][0]['visible'][i] = True
        step['args'][0]['visible'][i+num_slices] = True
        steps.append(step)

    sliders = [dict(
        active=11,
        currentvalue={"prefix": "Slice: "},
        #pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        width = 1170,
        height = 735,
        sliders=sliders
    )

    return fig

def compute_heatmap(pet_scan, model = model, layer_name = 'conv3d_3', eps = 1e-8):

    pet_scan = np.expand_dims(pet_scan, axis = 0)
    grad_model = tf.keras.Model(inputs = [model.inputs],
                                outputs = [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:

        inputs = tf.cast(pet_scan, tf.float32)

        (conv_outputs, prediction) = grad_model(inputs)
        loss = prediction[:,0]

    grads = tape.gradient(loss, conv_outputs)

    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads

    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]

    weights = tf.reduce_mean(guided_grads, axis = (0,1,2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis = -1)

    h_cam, w_cam, d_cam = (cam.shape[0], cam.shape[1], cam.shape[2])
    h, w, d = (pet_scan.shape[1], pet_scan.shape[2], pet_scan.shape[3])
    scales = (h/h_cam, w/w_cam, d/d_cam)

    heat_map = ndimage.zoom(cam.numpy(),
                            zoom = scales,
                            mode = 'nearest')

    heat_map = (heat_map - np.min(heat_map))/(np.max(heat_map) - np.min(heat_map) + eps)
    #heat_map = (heat_map * 255).astype('uint8')

    return heat_map
