import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import torch
import face_recognition
from PIL import Image


# loading input metadata
# input_path = r"C:\Users\omri_\Downloads\train_sample_videos/"
input_path = r"C:\Users\Bengal\Desktop\project/"

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


train_images = os.listdir(resized_output_path)
train_labels = [train_sample_metadata.loc[vid+".mp4"]['label'] for vid in [re.sub('[0-9]+.jpg','',item) for item in os.listdir(resized_output_path)]]
train_labels_df = pd.DataFrame(zip(train_images, train_labels), columns=["image", "label"])


def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


img = resized_output_path+train_labels_df.loc[0]["image"]
img = cv2.imread(img)
temp = grayConversion(img)
cv2.imshow("GrayScale", temp)

labels = (train_labels_df["label"] == 'REAL').apply(int)

train = []
for img in tqdm(resized_output_path+train_labels_df["image"]):
    img = cv2.imread(img)
    temp = grayConversion(img)
    train.append(temp)

train = np.array(train)
np.array(train).shape   # (7442,256,256)

train = torch.tensor(train).float()
train_labels = torch.tensor(labels)

acc_none, acc_none_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_None,optimizer_input= None, n_epochs = 15)






