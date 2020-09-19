import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm


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


all_image_names = [re.sub(".mp4","",img) for img in os.listdir(input_path) if re.findall(".mp4",img)]
for img_name in tqdm(all_image_names):
    VIDEO_STREAM = input_path+img_name+".mp4"
    vidcap = cv2.VideoCapture(VIDEO_STREAM)

    images_path = input_path+"images_"+img_name
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



