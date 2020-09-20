import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import face_recognition
from PIL import Image


# loading input metadata
input_path = r"C:\Users\omri_\Downloads\train_sample_videos/"
train_sample_metadata = pd.read_json(input_path+'metadata.json').T
train_sample_metadata.head()

# plotting dist of labels
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()


# func that captures images and saves them
def getFrame(vidcap, sec, count, path):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(path+"\image"+str(count)+".jpg", image)
    return hasFrames

# creating images from video at constant intervals
all_video_names = [re.sub(".mp4","",img) for img in os.listdir(input_path) if re.findall(".mp4",img)]
for video_name in tqdm(all_video_names):
    VIDEO_STREAM = input_path+video_name+".mp4"
    vidcap = cv2.VideoCapture(VIDEO_STREAM)

    images_path = input_path+"images_"+video_name
    os.makedirs(images_path, exist_ok=True)

    sec = 0
    frameRate = 0.5  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(vidcap, sec, count, images_path)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(vidcap, sec, count, images_path)

# capturing faces within the images
for video_name in tqdm(all_video_names):
    count = 1
    for i in range(1,21):
        images_path = input_path + "images_" + video_name + "/image"+str(i)+".jpg"
        src = cv2.imread(images_path)
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(image)

        face_images_path = input_path + "face_images_" + video_name
        os.makedirs(face_images_path, exist_ok=True)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            # plt.imshow(face_image)
            cv2.imwrite(face_images_path + r"\face_image" + str(count) + ".jpg", face_image)
            count += 1

# resizing images to 256x256
size = 256, 256
resized_output_path = input_path + "all_train_imgs/"
os.makedirs(resized_output_path, exist_ok=True)
for video_name in tqdm(all_video_names):
    face_images_path = input_path + "face_images_" + video_name
    for face_img in os.listdir(face_images_path):
        im = Image.open(face_images_path+"/"+face_img)
        im = im.resize(size, Image.ANTIALIAS)
        im_num = re.findall('[0-9]+', face_img)[0]
        im.save(resized_output_path+video_name+im_num+".jpg", "JPEG")

