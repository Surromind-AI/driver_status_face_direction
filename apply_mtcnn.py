import cv2
import os
import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from mtcnn import MTCNN
from tqdm import tqdm

def image_resize():
    train_0 = os.listdir("/home/djlee/data/eyes/train/0")
    train_1 = os.listdir("/home/djlee/data/eyes/train/1")
    test_0 = os.listdir("/home/djlee/data/eyes/test/0")
    test_1 = os.listdir("/home/djlee/data/eyes/test/1")

    for t0, t1 in zip(train_0, train_1):
        img1 = cv2.imread("/home/djlee/CPG/data/eyes/train/0/"+t0)
        img2 = cv2.imread("/home/djlee/CPG/data/eyes/train/1/"+t1)
        

        resize_img1 = cv2.resize(img1, (112, 112))
        resize_img2 = cv2.resize(img2, (112, 112))
        

        cv2.imwrite("/home/djlee/datasets/eyes_resize/train/0/"+t0, resize_img1)
        cv2.imwrite("/home/djlee/datasets/eyes_resize/train/1/"+t1, resize_img2)
        

    for tt0, tt1 in zip(test_0, test_1):
        img3 = cv2.imread("/home/djlee/CPG/data/eyes/test/0/"+tt0)
        img4 = cv2.imread("/home/djlee/CPG/data/eyes/test/1/"+tt1)

        resize_img3 = cv2.resize(img3, (112, 112))
        resize_img4 = cv2.resize(img4, (112, 112))

        cv2.imwrite("/home/djlee/datasets/eyes_resize/test/0/"+tt0, resize_img3)
        cv2.imwrite("/home/djlee/datasets/eyes_resize/test/1/"+tt1, resize_img4)

import torch
from torchvision import datasets, transforms
def mean_std():
    dataset = datasets.ImageFolder('/home/djlee/CPG/data/face_eyes/train',
                 transform=transforms.ToTensor())

    loader = torch.utils.data.DataLoader(dataset,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False)

    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    print("mean:", mean)
    print("std:", std)
    
def mtcnn():
    down_path = "/home/dataset/face_v0_1/down/"
    up_path = "/home/dataset/face_v0_1/up/"
    left_path = "/home/dataset/face_v0_1/left/"
    right_path = "/home/dataset/face_v0_1/right/"
    eyes_open_path = "/home/dataset/face_v0_1/eyes_open/"
    eyes_close_path = "/home/dataset/face_v0_1/eyes_close/"

    down_file_names = os.listdir(down_path)
    up_file_names = os.listdir(up_path)
    left_file_names = os.listdir(left_path)
    right_file_names = os.listdir(right_path)
    eyes_open_file_names = os.listdir(eyes_open_path)
    eyes_close_file_names = os.listdir(eyes_close_path)

    detector = MTCNN()
    with tqdm(total=len(down_file_names) + len(up_file_names) + len(left_file_names) + len(right_file_names) + len(eyes_open_file_names) + len(eyes_close_file_names)) as t:
        for down in down_file_names:
            try:
                file_name = "/home/dataset/broad_mtcnn/down/" + down
                img = cv2.cvtColor(cv2.imread(down_path+down), cv2.COLOR_BGR2RGB)
                box = detector.detect_faces(img)
                if not box:
                    continue
                box = box[0]["box"]

                img = img[box[1]-100:int(box[1])+int(box[3])+100, box[0]-100:int(box[0])+int(box[2])+100]
                cv2.imwrite(file_name, cv2.resize(img, (112, 112)))
                t.update(1)
            except:
                continue

        for up in up_file_names:
            try:
                file_name = "/home/dataset/broad_mtcnn/up/" + up
                img = cv2.cvtColor(cv2.imread(up_path+up), cv2.COLOR_BGR2RGB)
                box = detector.detect_faces(img)
                if not box:
                    continue
                box = box[0]["box"]

                img = img[box[1]-100:int(box[1])+int(box[3])+100, box[0]-100:int(box[0])+int(box[2])+100]
                cv2.imwrite(file_name, cv2.resize(img, (112, 112)))
                t.update(1)
            except:
                continue

        for left in left_file_names:
            try:
                file_name = "/home/dataset/broad_mtcnn/left/" + left
                img = cv2.cvtColor(cv2.imread(left_path+left), cv2.COLOR_BGR2RGB)
                box = detector.detect_faces(img)
                if not box:
                    continue
                box = box[0]["box"]

                img = img[box[1]-100:int(box[1])+int(box[3])+100, box[0]-100:int(box[0])+int(box[2])+100]
                cv2.imwrite(file_name, cv2.resize(img, (112, 112)))
                t.update(1)
            except:
                continue

        for right in right_file_names:
            try:
                file_name = "/home/dataset/broad_mtcnn/right/" + right
                img = cv2.cvtColor(cv2.imread(right_path+right), cv2.COLOR_BGR2RGB)
                box = detector.detect_faces(img)
                if not box:
                    continue
                box = box[0]["box"]

                img = img[box[1]-100:int(box[1])+int(box[3])+100, box[0]-100:int(box[0])+int(box[2])+100]
                cv2.imwrite(file_name, cv2.resize(img, (112, 112)))
                t.update(1)
            except:
                continue

        for eyes_open in eyes_open_file_names:
            try:
                file_name = "/home/dataset/broad_mtcnn/eyes_open/" + eyes_open
                img = cv2.cvtColor(cv2.imread(eyes_open_path+eyes_open), cv2.COLOR_BGR2RGB)
                box = detector.detect_faces(img)
                if not box:
                    continue
                box = box[0]["box"]

                img = img[box[1]-100:int(box[1])+int(box[3])+100, box[0]-100:int(box[0])+int(box[2])+100]
                cv2.imwrite(file_name, cv2.resize(img, (112, 112)))
                t.update(1)
            except:
                continue

        for eyes_close in eyes_close_file_names:
            try:
                file_name = "/home/dataset/broad_mtcnn/eyes_close/" + eyes_close
                img = cv2.cvtColor(cv2.imread(eyes_close_path+eyes_close), cv2.COLOR_BGR2RGB)
                box = detector.detect_faces(img)
                if not box:
                    continue
                box = box[0]["box"]

                img = img[box[1]-100:int(box[1])+int(box[3])+100, box[0]-100:int(box[0])+int(box[2])+100]
                cv2.imwrite(file_name, cv2.resize(img, (112, 112)))
                t.update(1)
            except:
                continue

mtcnn()