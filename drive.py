import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2
import math

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops


sio = socketio.Server()
app = Flask(__name__)
model = None
new_rows = 66 
new_cols = 200

# crop numbers, tried a few combinations, these seem to work better
c_row_start = 60
c_row_end = 140
c_col_start = 40
c_col_end = 280


def crop_resize_image(image):
    # crop
    cropped_image = image[c_row_start:c_row_end, c_col_start:c_col_end]

    # resize
    resized_image = cv2.resize(cropped_image, (new_cols, new_rows))

    return resized_image

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    x = np.asarray(image, dtype=np.float32)

    # crop per model input
    image_array = crop_resize_image(cv2.cvtColor(x, cv2.COLOR_RGB2YUV))
    transformed_image_array = image_array[None, :, :, :]

    # predict
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    speed = float(speed)
    angle_shift = 0.12 # Same as in my model

    # Slow down for turns
    if abs(steering_angle) > angle_shift:
        throttle = 0.1
    else:
        throttle = 0.2

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r', encoding="latin-1") as jfile:
        model = model_from_json(json.load(jfile))
        # model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)