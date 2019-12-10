# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io, os, zipfile, shutil

from detection import TrojanDetector2
from helper import image_trojan_setup_helper, model_inspector
from helper import setup_seed, setup_feed, train_callback
from loss import setup_loss1
import api

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config["UPLOAD_FOLDER"]= './tmp'

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

h = image_trojan_setup_helper()
ins = model_inspector()
    
metadata = { 
    "setup_input_feeds_callback"    : setup_seed,
    "customized_setup_input_tensors": h.inject_trojan_cifar,
    "customized_setup_inspect_model": ins.inspector1,
}
config = {
    "customized_setup_loss": setup_loss1,
    "train_callback" : train_callback
}

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = ResNet50(weights="imagenet")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/status", methods=["GET"])
def status():
	return api.status(flask)

@app.route("/display", methods=["GET"])
def display():
	return api.display(flask)

@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	
	seed_path = os.path.join("tmp", "seeds")
	metadata["setup_input_feeds_callback_args1"] = seed_path	
	
	if flask.request.method == "POST":
		print("[+] files:", flask.request.files)
		try:
			shutil.rmtree('tmp')
			os.makedirs('tmp')
			os.makedirs(seed_path)
		except Exception as e:
			print(e)
		
		if flask.request.files.get("model"):
			fmodel = flask.request.files["model"]
			target_model_path = os.path.join("tmp", fmodel.filename)
			fmodel.save(target_model_path)

		if flask.request.files.get("seeds"):
			fseed = flask.request.files["seeds"]
			zip_path = os.path.join("tmp", fseed.filename)
			fseed.save(zip_path)
			with zipfile.ZipFile(zip_path, 'r') as zip_ref:
				zip_ref.extractall(seed_path)
		else:
			metadata["seed_gen"] = seed_path
			metadata["setup_input_feeds_callback_args1"] = None
			
		with tf.variable_scope("cn1", reuse=tf.AUTO_REUSE):
			detector = TrojanDetector2(target_model_path, metadata, config)
			res = detector.detect()
		data["success"] = True
			
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0', debug = False, threaded = True)