import cv2
from keras.models import load_model
import keras
import numpy as np
import pandas as pd

model = load_model(r'resnetmodel.hdf5')
#Input dimension: (16, 96, 94, 3) (RGB video with 16 frames of size 96x64)
print("hey")
FRAME_SIZE = (96, 64)
N_FRAME = 16 
NORMALIZE_CONST = 255.0
TERMINATE_KEY = 'q'

labels = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
        "Pushing Hand Away", "Pulling Hand In", "Sliding Two Fingers Left", "Sliding Two Fingers Right", 
        "Sliding Two Fingers Down", "Sliding Two Fingers Up", "Pushing Two Fingers Away", "Pulling Two Fingers In",
        "Rolling Hand Forward", "Rolling Hand Backward", "Turning Hand Clockwise", "Turning Hand Counterclockwise",
        "Zooming In With Full Hand", "Zooming Out With Full Hand", "Zooming In With Two Fingers", "Zooming Out With Two Fingers",
        "Thumb Up", "Thumb Down", "Shaking Hand", "Stop Sign",
        "Drumming Fingers", "No Gesture", "Doing Other Things"]
idx_to_label = {i: labels[i] for i in range(27)}

#Capture the first and only camera
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
print("hey2")
labels = pd.read_csv(r'jester-v1-labels.csv', header= None)
buffer = []
predicte_value = 0
final_label = ""
i = 0
while (vid.isOpened()):
    ret,frame = vid.read()
    if ret:
        i += 1
        #resize to match training sample
        image = cv2.resize(frame,(FRAME_SIZE))
        #normalize
        image = image/NORMALIZE_CONST
        buffer.append(image)
        if(i% N_FRAME == 0): #one complete instance
            buffer = np.expand_dims(buffer,0)
            predicted_value = np.argmax(model.predict(buffer))
            final_label = idx_to_label[predicted_value]
            cv2.imshow('frame',frame)
            buffer = []
        text = "activity: {}".format(final_label)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5) 
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord(TERMINATE_KEY):
        break
vid.release()
cv2.destroyAllWindows()


