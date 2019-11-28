# server.py

import flask
from flask import Flask, request, jsonify, send_file, g, abort
import requests
import collections
from os import listdir
from os.path import isfile, join
import json
import imageio
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import code
import time
import cv2
import torchvision
import tifffile as tiff
from tifffile import imsave
from scipy import interpolate
import sys
import models


PORT = 8080

## The classifier variable assumes only one user 
## will access the server at a time to set the model
## and run predictions.
classifier = None
classifier_metadata = None

count_request = 0


available_json_models = {} # {'classifier1.json': {'pixel_classifier_type': 'PyTorchPixelClassifier', 'model':...}}
path_to_model_directory = "M:/deep_learning/q3/models/"
for models in listdir(path_to_model_directory):
    path_to_file = join(path_to_model_directory, models)
    if isfile(path_to_file) and path_to_file.endswith(".json"):
        with open(path_to_file) as json_file:
            data = json.load(json_file)
            available_json_models[models] = data
json_data = json.dumps(available_json_models)

print('Launching the server...')
app = flask.Flask(__name__)

    
@app.route('/checkStatus', methods=['GET'])
def check_status():
    return "OK"


@app.route('/getAvailableModels', methods=['POST'])
def get_available_models():
    return available_json_models


@app.route('/setModel', methods=['POST'])
def set_model():
    try:
        request.get_data()
        data = request.data
        path_to_classifier = available_json_models[data.decode() + ".json"]["model"]["pathModel"]
        
        global classifier
        global classifier_metadata
        
        if classifier != None:
            classifier.cpu()
        
        classifier = torch.load(str(path_to_classifier))
        classifier_metadata = available_json_models[data.decode() + ".json"]
        
        torch.cuda.empty_cache()
        classifier = classifier.eval()
        classifier = classifier.to('cuda:0')
        
        return "OK"
    except Exception as e:
        return str(e)


@app.route('/predict', methods=['POST'])
def predict():
    global classifier
    global classifier_metadata
    
    request.get_data()
    data = request.data
    f = BytesIO(data)
    im = tiff.imread(f)
    
    ## Normalise according to metadata
    im_mean = classifier_metadata["model"]["means"]
    im_std = classifier_metadata["model"]["stds"]
    im = (im - im_mean) / im_std
    
    ## If RGB, shape received was [width, height, channel]
    ## We want [channel, width, height]
    if im.shape[2] == 3:
        im = im.transpose(2, 0, 1)
        
    ## Works for only one channel for now
    if len(classifier_metadata["metadata"]["inputChannels"]) > 0:
        im = im[np.newaxis, classifier_metadata["metadata"]["inputChannels"][0], :, :]
    

    
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        print("CUDA error: an illegal memory access was encountered")
        print("Continue anyway.")
        
        
    input_tensor = torch.from_numpy(im).unsqueeze(0).float()
    try:
        with torch.no_grad():
            output = classifier(input_tensor.cuda())
    except:
        del input_tensor
        exception = sys.exc_info()[0]
        print('Could not get prediction from input tensor:' + str(exception))
        abort(400)
    
    output = output.cpu().data.numpy()
    del input_tensor
    torch.cuda.empty_cache()
    
    """
    ####################################################################
    ##########  This is only to print out the images, for debug purposes
    global count_request
    count_request += 1
    img_out = output.copy()
    img_out = np.argmax(img_out, axis=1)
    img_out[img_out > 0] = 255
    
    stacked_img = np.stack((np.squeeze(img_out),)*3, axis=-1)
    concat_img = np.concatenate((np.asarray(im), stacked_img), axis=1)
    cv2.imwrite("M:/deep_learning/out" + str(count_request) + ".png", concat_img)
    ####################################################################
    """
    
    #code.interact(banner="Start", local=dict(globals(), **locals()), exitmsg="End")

    if classifier_metadata["metadata"]["outputType"] == "CLASSIFICATION":
        if output.shape[1] == 1:
            byte_stream = BytesIO()
            imsave(byte_stream, output)
            byte_stream.seek(0)
            
            return send_file(byte_stream, mimetype='image/tiff')
        else:
            output = np.argmax(output, axis=1) ## Now shape = [channel, height, width]
            pil_image = Image.fromarray(output.squeeze().astype('uint8')) ## Now shape = [height, width]
            byte_stream = BytesIO()
            pil_image.save(byte_stream, 'TIFF', quality=100)
            byte_stream.seek(0)
            
            return send_file(byte_stream, mimetype='image/tiff')
    
    
    elif classifier_metadata["metadata"]["outputType"] == "PROBABILITY":
        output = np.squeeze(output)
        if classifier_metadata["doSoftMax"] == True:
            output = np.exp(output)
            output = (output/sum(output)).astype("float32")
        else:
            output = ((output - output.min()) * (1/(output.max() - output.min()) * 255)).astype("uint8")
        
        #for channel in range(output.shape[0]):
            #np.save("M:/deep_learning/out" + str(channel) + ".npy", output[channel, :, :])
        byte_stream = BytesIO()
        imsave(byte_stream, output)
        byte_stream.seek(0)
        
        return send_file(byte_stream, mimetype='image/tiff')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
    