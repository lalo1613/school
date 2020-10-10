import numpy as np
import pandas as pd
import cv2
import pickle
import os
import re
from tqdm import tqdm
import torch
#import face_recognition
from PIL import Image


def getFrame(vidcap, sec, count, path):
    # func that captures images and saves them
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(path+"\image"+str(count)+".jpg", image)
    return hasFrames


def pre_process_dataset(input_path, set_str):

    if set_str + "_torch.pkl" in os.listdir(input_path):
        with open(input_path + set_str + "_torch.pkl", "rb") as file:
            load_dict = pickle.load(file)

        dataset, dataset_labels, video_labels = load_dict["dataset"], load_dict["dataset_labels"], load_dict["video_belongings"]
        return dataset, dataset_labels, video_labels

    metadata = pd.read_json(input_path+'metadata.json').T

    # creating images from video at constant intervals
    all_video_names = [re.sub(".mp4","",img) for img in os.listdir(input_path) if re.findall(".mp4",img)]

    sec = 0
    frameRate = 0.5  # //it will capture image in each 0.5 second
    count = 1

    for video_name in tqdm(all_video_names):
        VIDEO_STREAM = input_path+video_name+".mp4"
        vidcap = cv2.VideoCapture(VIDEO_STREAM)

        images_path = input_path+"images_"+video_name
        os.makedirs(images_path, exist_ok=True)

        success = getFrame(vidcap, sec, count, images_path)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(vidcap, sec, count, images_path)

    # capturing faces within the images
    for video_name in tqdm(all_video_names):
        count = 1
        if "face_images_" + video_name not in os.listdir(input_path):
            for i in range(1, 21):
                images_path = input_path + "images_" + video_name + "/image"+str(i)+".jpg"
                src = cv2.imread(images_path)
                if src is not None:
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
    size = 32, 32
    resized_output_path = input_path + "new_all_"+set_str+"_imgs/"
    os.makedirs(resized_output_path, exist_ok=True)
    for video_name in tqdm(all_video_names):
        face_images_path = input_path + "face_images_" + video_name
        for face_img in os.listdir(face_images_path):
            im = Image.open(face_images_path+"/"+face_img)
            im = im.resize(size, Image.ANTIALIAS)
            im_num = re.findall('[0-9]+', face_img)[0]
            im.save(resized_output_path+video_name+im_num+".jpg", "JPEG")

    set_images = os.listdir(resized_output_path)
    set_labels = [metadata.loc[vid+".mp4"]['label'] for vid in [re.sub('[0-9]+.jpg','',item) for item in os.listdir(resized_output_path)]]
    set_labels_df = pd.DataFrame(zip(set_images, set_labels), columns=["image", "label"])

    dataset = []
    for img in tqdm(resized_output_path+set_labels_df["image"]):
        img = cv2.imread(img)
        img = np.moveaxis(img, 2, 0)
        dataset.append(img)
    dataset = np.array(dataset)

    # np.array(dataset).shape   # (7442,256,256)
    dataset = torch.tensor(dataset).float()
    dataset_labels = (set_labels_df["label"] == "REAL").apply(int)
    dataset_labels = torch.tensor(dataset_labels)
    video_names = set_labels_df["image"].apply(lambda x: re.sub("[0-9]+\.jpg","",x))

    with open(input_path+set_str+"_torch.pkl","wb") as file:
        pickle.dump({"dataset":dataset, "dataset_labels": dataset_labels, "video_belongings": video_names}, file)

    return dataset, dataset_labels, video_names


# # loading test_input data
# train_input_path = r"C:\Users\omri_\Downloads\train_videos/"
# test_input_path = r"C:\Users\omri_\Downloads\train_sample_videos/"
# # test_input_path = r"C:\Users\Bengal\Desktop\project/"
#
# train_set, train_labels = pre_process_dataset(train_input_path, "train")
# test_set, test_labels = pre_process_dataset(test_input_path, "test")

#
# def grayConversion(image):
#     grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
#     gray_img = grayValue.astype(np.uint8)
#     return gray_img
#
#
# img = cv2.imread(r"C:\Users\omri_\Downloads\train_videos\face_images_anzenqcwqo/face_image5.jpg")
# temp = grayConversion(img)
# cv2.imwrite(r"C:\Users\omri_\Downloads/gray_example.jpg", temp)
