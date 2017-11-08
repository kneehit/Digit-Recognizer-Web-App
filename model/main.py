import logging
from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
import keras.models
import re
import sys 

import os
sys.path.append(os.path.abspath("./model"))
from load import * 
app = Flask(__name__)


global model, graph
model, graph = init()


def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)

	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))
	

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	
	imgData = request.get_data()

	convertImage(imgData)
	print "debug"

	x = imread('output.png',mode='L')

	x = np.invert(x)

	x = imresize(x,(28,28))
	#imshow(x)

	x = x.reshape(1,28,28,1)
	print "debug2"

	with graph.as_default():

		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print "debug3"
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
    
   
if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)

 
# [END app]