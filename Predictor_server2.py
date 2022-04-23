import socket
import sys
import json
import os
import tensorflow as tf
from tensorflow import keras
from os import listdir
from res_model import se_resnet50, se_resnet50_gpc_in_chain_only
import cv2 
import numpy as np

#Use gpu
print(tf.__version__)
gpu = tf.config.list_physical_devices('GPU')[0]
print(gpu)
tf.config.experimental.set_memory_growth(gpu, True)

# configuration
ICLASS = 'yeast'
N_CLASSES = 2
H, W, C = 200, 200, 3
RESCALE = True
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = EPOCHS // 5
UNITS = [64, 128, 256, 512]
LOOPS = [3, 4, 6, 3, 2]
LR = 0.0005
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)

# Load model
new_model_GNB = se_resnet50(in_shape=(H, W, C), n_classes = N_CLASSES, opt=OPTIMIZER, units = UNITS, loops = LOOPS, dropout_rate=0)
new_model_GPB = se_resnet50(in_shape=(H, W, C), n_classes = N_CLASSES, opt=OPTIMIZER, units = UNITS, loops = LOOPS, dropout_rate=0)
new_model_GPC = se_resnet50(in_shape=(H, W, C), n_classes = N_CLASSES, opt=OPTIMIZER, units = UNITS, loops = LOOPS, dropout_rate=0)
gpc_classifier = r'C:\Users\Hsulab32\jonah\best_model\GPC\GPC_3_20220418_123050'
new_model_chainPredict = se_resnet50_gpc_in_chain_only(gpc_classifier=gpc_classifier,in_shape=(H, W, C), n_classes = N_CLASSES, opt=OPTIMIZER, units = UNITS, loops = LOOPS, dropout_rate=0)
new_model_Yeast = se_resnet50(in_shape=(H, W, C), n_classes = N_CLASSES, opt=OPTIMIZER, units = UNITS, loops = LOOPS, dropout_rate=0)

# Checkpoint
GNB_checkpoint = r'C:\Users\Hsulab32\jonah\best_model\GNB\GNB_5_20220415_202803'
GPB_checkpoint = r'C:\Users\Hsulab32\jonah\best_model\GPB\GPB_7_20220417_152318'
GPC_checkpoint = r'C:\Users\Hsulab32\jonah\best_model\GPC\GPC_3_20220418_123050'
GPCinchain_checkpoint = r'C:\Users\Hsulab32\jonah\best_model\GPC-in-chain\GPC-in-chain_3_20220420_162712'
yeast_checkpoint = r'C:\Users\Hsulab32\jonah\best_model\yeast\yeast_4_20220415_140145'

# Load weight
new_model_GNB.load_weights(GNB_checkpoint)
new_model_GPB.load_weights(GPB_checkpoint)
new_model_GPC.load_weights(GPC_checkpoint)
new_model_chainPredict.load_weights(GPCinchain_checkpoint)
new_model_Yeast.load_weights(yeast_checkpoint)

# Trainable false
new_model_GNB.trainable = False
new_model_GPB.trainable = False
new_model_GPC.trainable = False
new_model_chainPredict.trainable = False
new_model_Yeast.trainable = False

# Set model compile
new_model_GNB.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
new_model_GPB.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
new_model_GPC.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
new_model_chainPredict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
new_model_Yeast.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

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
                if img_data is None: 
                    print("no image data")
                    break
                else:
                    print("there image data")
                    cv2.imwrite('tst.jpg', img_data)
                
            except socket.error:
                print("Error Occured.")
    
            finally:
                if not os.path.exists("tst.jpg"):break
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
            if not os.path.exists("tst.jpg"):
                connection.send(("no data").encode())
                break;
            # Predict and send the predicted result to client
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
        if os.path.exists("tst.jpg"): os.remove('tst.jpg')
        connection.close()
        print("finish connection")