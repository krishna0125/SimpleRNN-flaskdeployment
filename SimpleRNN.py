import cv2 as cv
from skimage.color import rgb2gray
import numpy as np
import os
from keras.models import load_model 
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,precision_score
import pickle
from keras import backend as K
from keras_pickle_wrapper import KerasPickleWrapper

path="C:/Users/Selvamani/Desktop/projects/Debris/"

files=os.listdir(path)

images=[]
original=[]
for i in range(len(files)):
    img=cv.imread(path+files[i])
    gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orig=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    images.append(gray)
    original.append(orig)
def gray_1(img):
    b=np.dot(img[:,:,:3],[0.21,0.5,0.7])
    return(b)
w_gray_img = gray_1(original[2])
ret,thresh4 = cv.threshold(w_gray_img,100,255,cv.THRESH_BINARY_INV)
x=w_gray_img.flatten()/255
y=thresh4.flatten()/255


##Image Format for Train
step = 2
# add step elements into train and test
train = np.append(x,np.repeat(x[-1,],step))



def convertToMatrix(data, step):
    x =[] 
    for i in range(len(data)-step):
        d=i+step  
        x.append(data[i:d,])
    return np.array(x)

trainX=convertToMatrix(train,step)

trainY=y

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)



model = Sequential()
model.add(SimpleRNN(units=30, input_shape=(1,step), activation="relu"))
model.add(Dense(8, activation="relu")) 
model.add(Dense(1,activation="sigmoid"))
model.compile(loss=dice_coef_loss, optimizer='rmsprop', metrics=[dice_coef])
model.summary()



model.fit(trainX,trainY, epochs=10, batch_size=100)


##Saving and Serializing

model_json= model.to_json()

model.save(r"C:/Users/Selvamani/Desktop/model/graph_weights_SRNN.h5")
with open("C:/Users/Selvamani/Desktop/model/SimpleRNN.json","w") as json_file:
    json_file.write(model_json)

##Saving Weights

model.save_weights("C:/Users/Selvamani/Desktop/model/Weights.h5")
