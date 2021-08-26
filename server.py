# server.py

import flask
from flask import request, jsonify, send_file, abort
import json
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torchvision
from tifffile import imsave
import sys
import importlib.util
from torchvision import transforms

PORT = 8080

# The classifier variable assumes only one user
# will access the server at a time to set the model
# and run predictions.
classifier = None

count_request = 0

print('Launching the server...')
app = flask.Flask(__name__)


@app.route('/checkStatus', methods=['GET'])
def check_status():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/setModelStructure', methods=['POST'])
def set_model_structure():
    try:
        request.get_data()
        data = request.data

        # Trying to import from file
        path = data.decode()
        name = path[path.rindex("/") + 1:len(path)]

        try:
            importlib.util.spec_from_file_location(name, path).loader.load_module()
        except Exception as e:
            global classifier
            classifier = torch.load(path)
        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    except FileNotFoundError:
        abort(404, description="Resource not found")  # Could technically be something different?


@app.route('/setModelWeights', methods=['POST'])
def set_model_weights():
    request.get_data()
    path_to_weights = request.data.decode()
    global classifier

    if classifier is not None:
        classifier.cpu()

    try:
        classifier = torch.load(path_to_weights)
        torch.cuda.empty_cache()
        classifier = classifier.eval()
        classifier = classifier.cuda()
    except FileNotFoundError as e:
        exception = sys.exc_info()[0]
        print('Could not load model weights: ' + str(e))
        abort(400, description=e)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/setPyTorchModel', methods=['POST'])
def set_pytorch_model():
    request.get_data()
    model_name = request.data.decode()
    global classifier

    if classifier is not None:
        classifier.cpu()

    try:
        classifier = getattr(torchvision.models, model_name)(pretrained=True)
        torch.cuda.empty_cache()
        classifier = classifier.eval()
        classifier = classifier.cuda()
    except Exception as e:
        print('Could not load PyTorch model: ' + str(e))
        abort(400, description=e)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/segment', methods=['POST'])
def segment():
    global classifier

    request.get_data()
    data = request.data
    f = BytesIO(data)
    im = np.array(Image.open(f))

    # TODO: Check with non-RGB images
    # If RGB, shape received was [w, h, c], but we want [c, w, h]
    if im.shape[2] == 3:
        im = im.transpose(2, 0, 1)

    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        print("CUDA error: an illegal memory access was encountered")
        print("Continue anyway.")

    # TODO: Using float here, maybe another dtype could be used depending on model..
    input_tensor = torch.from_numpy(im).unsqueeze(0).float()
    try:
        with torch.no_grad():
            output = classifier(input_tensor.cuda())
    except Exception as e:
        del input_tensor
        print('Could not get prediction from input tensor: ' + str(e))
        abort(400)

    output = output.cpu().detach().numpy()
    del input_tensor
    torch.cuda.empty_cache()

    if output.shape[1] == 1:
        print("1 dimension output")
        byte_stream = BytesIO()
        imsave(byte_stream, output)
        byte_stream.seek(0)

        del output
        return send_file(byte_stream, mimetype='image/tiff')
    else:
        output = np.argmax(output, axis=1)  # Now shape = [channel, height, width]

        pil_image = Image.fromarray(output.squeeze().astype('uint8'))  # Now shape = [height, width]
        byte_stream = BytesIO()
        pil_image.save(byte_stream, 'TIFF', quality=100)
        byte_stream.seek(0)
        
        del output
        return send_file(byte_stream, mimetype='image/tiff')


@app.route('/classify', methods=['POST'])
def classify():
    global classifier

    request.get_data()
    data = request.data
    f = BytesIO(data)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    im = transform(Image.open(f))

    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        print("CUDA error: an illegal memory access was encountered")
        print("Continue anyway.")

    input_tensor = im.unsqueeze(0)
    try:
        with torch.no_grad():
            output = classifier(input_tensor.cuda())
    except Exception as e:
        del input_tensor
        print('Could not get prediction from input tensor: ' + str(e))
        abort(400)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    probs = probs.cpu().detach().numpy()

    del input_tensor
    del output
    torch.cuda.empty_cache()

    # TODO: find a way to communicate the dtype of probabilities (here float32)
    response = flask.make_response(probs.astype(">f4").tobytes())
    response.headers.set('Content-Type', 'application/octet-stream')  # So it isn't considered text data
    return response


@app.route('/detect', methods=['POST'])
def detect():
    global classifier

    request.get_data()
    data = request.data
    f = BytesIO(data)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(Image.open(f)).unsqueeze(0)

    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        print("CUDA error: an illegal memory access was encountered")
        print("Continue anyway.")

    try:
        with torch.no_grad():
            classifier = classifier.eval()
            classifier = classifier.cuda()
            output = classifier(input_tensor.cuda())
    except Exception as e:
        del input_tensor
        print('Could not get prediction from input tensor: ' + str(e))
        abort(400)

    data = {'boxes': output[0]['boxes'].cpu().detach().numpy().tolist(),
            'labels': output[0]['labels'].cpu().detach().numpy().tolist(),
            'scores': output[0]['scores'].cpu().detach().numpy().tolist()}
    del input_tensor
    del output
    return jsonify(data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
