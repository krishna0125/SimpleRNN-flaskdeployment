from flask import Flask, render_template,request
#from scipy.misc import imsave, imread, imresize
import numpy as np
import matplotlib.pyplot as plt
from keras.backend import set_session
import keras.models
import re 
import os
import sys
import cv2 as cv
#from Debris1 import * 
#global model, graph
import tensorflow as tf
from tensorflow.keras import backend as K
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
sess = tf.Session()
sys.path.append(os.path.abspath("C:/Users/Selvamani/Desktop/model"))
people_folder="C:/Users/Selvamani/Desktop/model"

smooth=1
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['UPLOAD_FOLDER'] = people_folder
#model, graph = init()
#Model Function
def dice_coef(y_true, y_pred):
	keras.backend.get_session().run(tf.local_variables_initializer())
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model=keras.models.load_model(r"C:\Users\Selvamani\Desktop\model\graph_weights_SRNN.h5",
	                              custom_objects={'dice_coef': dice_coef,"dice_coef_loss":dice_coef_loss})
model._make_predict_function()
graph = tf.get_default_graph()

    
    
    
    




def gray_1(img):
    b=np.dot(img[:,:,:3],[0.2,0.5,0.9])
    return(b)

print("Gray Scale")
def convertToMatrix(data, step):
	X=[]
	for i in range(len(data)-step):
  		d=i+step  
  		X.append(data[i:d,])
	return np.array(X)
print("Grey Scale")
def pre_process(image):
	w_gray=gray_1(image)
	a=image.shape[0]
	b=image.shape[1]
	x=w_gray.flatten()/255
	step = 2
# add step elements into train and test
	test = np.append(x,np.repeat(x[-1,],step))
	print(test.shape)
	test=convertToMatrix(test,step)
	print(test.shape)
	test=np.reshape(test, (test.shape[0], 1, test.shape[1]))
	print(test.shape)
	return (test,a,b)
basepath="C:/Users/Selvamani/Desktop/projects/Debris"
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('image_classifier.html')




@app.route("/predict",methods=["GET","POST"])
def predict():

	if request.method == 'POST':
	    f = request.files['image']
	    basepath = os.path.dirname(__file__)
	    file_path = os.path.join(
	    basepath, secure_filename(f.filename))
	    f.save(file_path)
	    print(file_path)

        
        


        # Save the file to ./uploads
        
        
    
	img=cv.imread(file_path)
	print("success1")
	processed_image,a,b=pre_process(img)
	global graph
	with graph.as_default():
		out=model.predict(processed_image)


	#summ=sum(out)
	#	percentage=(summ/2354176)*100

	#KB="We Know How to Do it"+str(percentage)
	seg_img=out.reshape(a,b)
	fig = plt.figure()
	plt.title('Segmented Image')
	plt.imshow(seg_img,cmap="gray")
	fig.savefig("C:/Users/Selvamani/Desktop/model/static/seg_img11.png")
	#full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'seg_img.png')
	#print(full_filename)
	return render_template("index.html")
	
if __name__ == '__main__':
	app.run(debug = True)








