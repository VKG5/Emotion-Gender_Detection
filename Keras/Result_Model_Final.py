import numpy as np
import argparse
import cv2
import requests
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# For drawing the emoticons
from image_commons import nparray_as_image, draw_with_alpha

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# emotions will be displayed on your face from the webcam feed
model.load_weights('model5.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# To load the emoticons emotions
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
emoticons = [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]

# start the webcam feed
print("Start")

url = "http://10.12.139.207:8080/shot.jpg"

#cap = cv2.VideoCapture(0)
while True:

    img_resp = requests.get(url)    
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr, -1)

    img = cv2.flip(img, 1)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    
    # For making the bounding box
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # For choosing the emoticon
        new_predict = int(np.argmax(prediction))
        # print(new_predict)

        # To draw the emoticons
        image_to_draw = emoticons[new_predict]
        
        # For changing the position of the emoticon
        x -= 50
        y -= 90
        w -= 40
        h -= 40
        image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
        image_array = image_as_nparray(image_to_draw)
        for c in range(0, 3):
            source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                                + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)

    cv2.imshow('Output', cv2.resize(img,(1280,720),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1)==27:
        break

print("Done")
cv2.destroyAllWindows()