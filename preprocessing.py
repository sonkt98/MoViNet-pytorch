import os
import shutil

import cv2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

import json
def center_crop(im):
    width, height = im.size  # Get dimensions

    left = round((width - 1000) / 2)
    top = round((height - 1000) / 2)
    x_right = round(width - 1000) - left
    x_bottom = round(height - 1000) - top
    right = width - x_right
    bottom = height - x_bottom

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def imageCrop():
    target_dir = "/home/petpeotalk/MoViNet-pytorch/aihub_data/rawData1"
    save_dir = "/home/petpeotalk/MoViNet-pytorch/temp_data"
    for i, dir in enumerate(tqdm(os.listdir(target_dir))):
        dir_path = os.path.join(target_dir, dir)
        if not os.path.exists(os.path.join(save_dir, dir)):
            os.makedirs(os.path.join(save_dir, dir))
        else:
            continue
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            image = Image.open(file_path)
            cropped_image = center_crop(image)
            cropped_image.save(os.path.join(save_dir, dir, file))
# /home/petpeotalk/MoViNet-pytorch/temp_data/dog-footup-102126
# /home/petpeotalk/MoViNet-pytorch/temp_data/dog-footup-102183
# /home/petpeotalk/MoViNet-pytorch/temp_data/dog-footup-101251

def makeVideo():
    target_dir = "/home/petpeotalk/MoViNet-pytorch/aihub_data"
    save_dir = "/home/petpeotalk/MoViNet-pytorch/video_custom_data"
    for i, dir in enumerate(tqdm(os.listdir(target_dir))):
        label = dir.split('-')[-2]
        dir_path = os.path.join(target_dir, dir)
        save_dir_path = os.path.join(save_dir, label)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        save_file_path = os.path.join(save_dir_path, dir + ".mp4")
        command = "/usr/bin/ffmpeg -y -framerate 10 -pattern_type glob -i \'" +\
                  dir_path + "/*.jpg\' -c:v libx264 " + save_file_path + " -loglevel error"
        result = os.system(command)
        if result != 0:
            error_message = 'ffmpeg error (merge_frames)'
            raise Exception(error_message)


def create_new_video(save_path, video_name, image_array):
    (h, w) = image_array[0].shape[:2]
    if len(video_name.split('/')) > 1:
        video_name = video_name.split('/')[1]
    else:
        video_name = video_name.split('.mp4')[0]
        video_name = video_name + '.avi'
    save_video_path = os.path.join(save_path, video_name)
    output_video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (w, h), True)
    for frame in range(len(image_array)):
        output_video.write(image_array[frame])
    output_video.release()
    # cv2.destroyAllWindows()

def set_transforms(mode):
    if mode == 'train':
        transform = transforms.Compose(
            [transforms.Resize(256),  # this is set only because we are using Imagenet pre-train model.
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
             ])
    elif mode == 'test' or mode == 'val':
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))])
    return transform

def main_procesing_data(args, folder_dir, sampled_video_file=None, processing_mode='main'):
    """"
       Create the sampled data video,
       input - video, full length.
       function - 1. Read the video using CV2
                  2. from each video X (args.sampling_rate) frames are sampled reducing the FPS by args.sampling_rate (for example from 25 to 2.5 FPS)
                  3. The function randomly set the start point where the new sampled video would be read from, and Y(args.num_frames_to_extract) continues frames are extracted.
                  4. if processing_mode == 'main' The Y continues frames are extracted and save to a new video if not the data in tensor tyoe mode is passed to the next function
       Output: videos in length of X frames
       """
    if args.dataset == 'aihub':
        video_list = args.row_data_dir


def aihub_capture_and_sample_video(row_data_dir, video_name, num_frames_to_extract, sampling_rate, save_path,
                             processing_mode):
    # ====== setting the video to start reading from the frame we want ======
    image_array = []
    if processing_mode == 'live':
        transform = set_transforms(mode='test')
    image_list = os.listdir(row_data_dir)
    video_height = 1280
    video_width = 720
    # dumpImageSize=int((len(image_list)-(sampling_rate*num_frames_to_extract))/2)
    for i in range(len(image_list)):
        image_path = os.path.join(row_data_dir, image_list[i])
        image = cv2.imread(image_path)
        video_height, video_width, _ = image.shape
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if processing_mode == 'live' else image
        image = Image.fromarray(RGB_img.astype('uint8'), 'RGB')
        if processing_mode == 'live':
            image_array += [transform(image)]
        else:
            image_array += [np.uint8(image)]
    if processing_mode == 'main':
        create_new_video(save_path, video_name, image_array)
    return image_array, [video_width, video_height]

def checkLocation(file):
    with open(file, 'r') as f:
        json_data = json.load(f)
    if json_data['metadata']['location'] == "실내":
        return True
    else: return False

import csv
import pandas as pd

if __name__ == '__main__':
    # imageCrop()
    # makeVideo()

    # root="/home/petpeotalk/AIHUB/hackerton-example-dataset/train"
    # num_frames_to_extract=24
    # sampling_rate=1
    # for dir in os.listdir(root):
    #     video_list = os.path.join(root, dir)
    #     new_video_dir = os.path.join("/home/petpeotalk/AIHUB/hackerton-example-dataset/train_video", dir)
    #     if not os.path.exists(new_video_dir):
    #         os.makedirs(new_video_dir)
    #     print(len(os.listdir(video_list)))
    #     for i, video in enumerate(tqdm(os.listdir(video_list))):
    #         video_path = os.path.join(video_list, video)
    #         if (len(os.listdir(video_path))<10): print(video_path)
    #         if (os.path.exists(os.path.join(new_video_dir, video+".avi"))): continue
    #
    #         aihub_capture_and_sample_video(video_path, video + ".mp4", num_frames_to_extract,
    #                                        sampling_rate, new_video_dir, "main")

    # data = pd.read_csv('/home/petpeotalk/Downloads/hackerton_all_metadata.csv')
    #
    # for i, row in tqdm(data.iterrows()):
    #     video_list = "/home/petpeotalk/AIHUB/Validation/원천데이터/DOG"
    #     video = row['file_video']
    #     label = row['label']
    #     if row['location'] == '실내':
    #         video_path = os.path.join(video_list, video)
    #         print(video_path)
    #         save_dir_path = os.path.join("/home/petpeotalk/AIHUB/Valid_분류작업", row['worker'], label)
    #         if not os.path.exists(save_dir_path):
    #             os.makedirs(save_dir_path)
    #         save_file_path = os.path.join(save_dir_path, video)
    #         if os.path.exists(video_path):
    #             print(os.path.join(video_list, video))
    #             shutil.move(video_path, save_file_path)

    # data = pd.read_csv('/home/petpeotalk/Downloads/hackerton_metadata.csv')
    #
    # videos_list = []
    # for i in range(1, 7):
    #     videos = "/home/petpeotalk/AIHUB/Training/원천데이터_"+str(i)
    #     videos_list = os.listdir(videos)
    # for v in tqdm(os.listdir(videos)):
    #     name=""
    #     l=v.split('-')
    #     del l[0]
    #     name = "-".join(l)
    #
    #     name = name.split('.')[0]
    #     # print(name)
    #     # print(data.index[data['file_video'].str.contains(name)].tolist())
    #     if (len(data.index[data['file_video'].str.contains(name)].tolist())!=0):
    #         row=data[data['file_video'].str.contains(name)]
    #         # print(row)
    #         video_list = "/home/petpeotalk/AIHUB/Training/원천데이터_" + row['folder'].item().split('_')[-1]
    #         label_path = "/home/petpeotalk/AIHUB/Training/" + row['folder'].item()
    #         # label_path = "/home/petpeotalk/AIHUB/Training/라벨데이터"
    #         video = v
    #
    #         # check = checkLocation(os.path.join(label_path, video + ".json"))
    #         label = row['label'].item()
    #         if row['location'].item() == '실내':
    #             video_path = os.path.join(video_list, video)
    #             save_dir_path = os.path.join("/home/petpeotalk/AIHUB/Train_분류작업", row['worker'].item(), label)
    #             if not os.path.exists(save_dir_path):
    #                 os.makedirs(save_dir_path)
    #             save_file_path = os.path.join(save_dir_path, video)
    #             if os.path.exists(video_path):
    #                 print(os.path.join(video_list, video))
    #                 shutil.move(video_path, save_file_path)

    video_list = "/home/petpeotalk/AIHUB/Validation/원천데이터/DOG"
    new_video_dir = "/home/petpeotalk/AIHUB/seperate"

    for i, dir in enumerate(tqdm(os.listdir(video_list))):
        label = dir.split('-')[-2]
        dir_path = os.path.join(video_list, dir)
        if checkLocation(os.path.join("/home/petpeotalk/AIHUB/Validation/라벨링데이터/DOG",  dir + ".json")):
            save_dir_path = os.path.join("/home/petpeotalk/AIHUB/라벨", label)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            save_file_path = os.path.join(save_dir_path, dir)
            if not os.path.exists(save_file_path):
                shutil.move(dir_path, save_file_path)
