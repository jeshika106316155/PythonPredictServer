import socket
import sys
import json
import os
from warnings import catch_warnings
import tensorflow as tf
from tensorflow import keras
from os import listdir
#from matplotlib import image
import cv2 
import numpy as np

#Use gpu
print(tf.config.list_physical_devices())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

with tf.device(tf.DeviceSpec(device_type="GPU")):
    #Load model
    new_model_GNB = tf.keras.models.load_model('C:/Users/Hsulab32/Downloads/PredictPyhtonServer/BestModel/GNB_2021-05-31 18_45_45.280569.h5')
    new_model_GPB = tf.keras.models.load_model('C:/Users/Hsulab32/Downloads/PredictPyhtonServer/BestModel/GPB_2021-05-29 17_02_54.403106.h5')
    new_model_GPC = tf.keras.models.load_model('C:/Users/Hsulab32/Downloads/PredictPyhtonServer/BestModel/New GPC_2021-05-30 04_17_07.039256.h5')
    new_model_chainPredict = tf.keras.models.load_model('C:/Users/Hsulab32/Downloads/PredictPyhtonServer/BestModel/New GPC-in chain_2021-06-01 04_07_08.012464.h5')
    new_model_Yeast = tf.keras.models.load_model('C:/Users/Hsulab32/Downloads/PredictPyhtonServer/BestModel/yeast_2021-06-02 22_46_02.189740.h5')

    new_model_GNB.trainable = False
    new_model_GPB.trainable = False
    new_model_GPC.trainable = False
    new_model_chainPredict.trainable = False
    new_model_Yeast.trainable = False
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to the port
    server_address = ('203.72.73.18', 10000) #203.72.73.18
    print('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    # Listen for incoming connections
    sock.listen(1)

    while True:
        # Wait for a connection
        print('waiting for a connection')
        connection, client_address = sock.accept()
        try:
            print('connection from', client_address)
            
            while True:
                try:
                    data = connection.recv(300000) #193018,191144
                    print('received')
                    if not data or len(data)==0:
                        print('no data')
                        break
                    
                    image = np.asarray(bytearray(data))
                    # use imdecode function
                    img_data = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    cv2.imwrite('tst.jpg', img_data)
                    
                finally:
                    #read and crop image to 200 x 200
                    imgs=[]
                    img_data = cv2.imread("tst.jpg")
                    print(img_data.shape);
                    y=0
                    x=0
                    h=200
                    w=200
                    for h2 in range(img_data.shape[0]//h):
                        for w2 in range(img_data.shape[1]//w):
                            y=h2*200
                            x=w2*200
                            crop_img = img_data[y:y+h, x:x+w]
                            crop_img = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_AREA)
                            crop_imgExDims = crop_img / 255 #normalization
                            imgs.append(crop_imgExDims)
                            
                            cv2.imwrite('C:/Users/Hsulab32/Downloads/PredictPyhtonServer/CropImg/'+str(h2)+'_'+str(w2)+'.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                imgs=np.array(imgs, dtype='float32')
                print(imgs.shape)
                predict=new_model_GNB.predict(imgs, batch_size=48)
                predict=predict.tolist()
                resGNB=[result.index(max(result)) for result in predict]
                predict=new_model_GPB.predict(imgs, batch_size=48)
                predict=predict.tolist()
                resGPB = [result.index(max(result)) for result in predict]
                predict=new_model_GPC.predict(imgs, batch_size=48)
                predict=predict.tolist()
                resGPC = [result.index(max(result)) for result in predict]
                predict=new_model_chainPredict.predict(imgs, batch_size=48)
                predict=predict.tolist()
                resChain = [result.index(max(result)) for result in predict]
                predict=new_model_Yeast.predict(imgs, batch_size=48)
                predict=predict.tolist()
                resYeast = [result.index(max(result)) for result in predict]
                data = json.dumps({"GNB": resGNB, "GPB": resGPB, "GPC": resGPC, "GPCinChain":resChain, "Yeast":resYeast})
                connection.send(data.encode())
                
            print('received, yay!') 
                
        finally:
            connection.close()
            print("finish connection")