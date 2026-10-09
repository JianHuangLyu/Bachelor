from __future__ import print_function
import torch.utils.data as data
import torch
import glob
import cv2
import numpy as np
from PIL import Image
import random
from torchvision import transforms

# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
# import pandas as pd
transform1 = transforms.Compose([
	# transforms.CenterCrop((WIDTH,HEIGHT)), # 只能对PIL图片进行裁剪
	transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
	]
)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

TXT_EXTENSIONS = [
    '.txt'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)
def get_sub_num(path):
    filename = glob.glob(path+'/*/*')
    filename = [f.replace("\\", "/") for f in filename]
    sublist = []
    # filelist = []
    # for file in filename:
    #     if str.find(file, 'flow_x_CASME2') > 0:
    #         if is_image_file(file):
    #             filelist.append(file)
    # only save x_flow files in self.filename
    # filename = filelist
    
    for i in range(len(filename)):
        if str.find(filename[i], 'flow_x_') > 0:
            sublist.append(
                filename[i].split('/')[-1].split('_')[2] + '_' + filename[i].split('/')[-1].split('_')[3]+'_'+ filename[i].split('/')[-1].split('_')[4])

    sublist_quchong = list(set(sublist))
    # sublist_quchong.sort()
    sub_num = len(sublist_quchong)
    print('Total subjects in the dataset:', sub_num)
    return len(sublist_quchong)

class MyDataset(data.Dataset):
    def __init__(self, path, mode, testsub, model='3d', width=28, class_num=4, augment=None):
        self.train_sub = []
        self.val_sub = []
        self.test_sub = []
        self.testsub = testsub
        self.path = path
        self.image_width = width
        self.augment = None
        self.mode = mode
        self.model = model
        self.split_num_x = 4
        self.num_class = class_num

        filename = glob.glob(path+'/*/*')
        filename = [f.replace("\\", "/") for f in filename]

        #only save x_flow files in self.filename
        self.filename = filename
        self.mode = mode
        #get subjectlist
        self.create_file_list()
        [train_sample_list, train_label_list,\
         self.val_sample_list, val_label_list,\
         self.test_sample_list, test_label_list, self.test_sample_name_list] = self.read_sample_label_list()
        self.train_y_data = torch.LongTensor(np.array(train_label_list))
        self.val_y_data = torch.LongTensor(np.array(val_label_list))
        self.test_y_data = torch.LongTensor(np.array(test_label_list))
        self.train_sample_list = train_sample_list
        self.train_label_list = train_label_list
        del train_label_list, val_label_list, test_label_list, filename

    def __getitem__(self, index): #返回tentor
        if (self.mode == 'train'):
            self.imgs = self.read_flow_data(self.train_sample_list[index])
            return self.imgs, self.train_y_data[index]
        if (self.mode == 'val'):
            self.imgs = self.read_flow_data(self.val_sample_list[index])
            return self.imgs, self.val_y_data[index]
        if (self.mode == 'test'):
            self.imgs = self.read_flow_data(self.test_sample_list[index])
            return self.imgs, self.test_y_data[index]

    def __len__(self):
        if (self.mode == 'train'):
            # print(self.train_sample_list.shape[0])
            return self.train_sample_list.shape[0]
        if (self.mode == 'val'):
            return self.val_sample_list.shape[0]
        if (self.mode == 'test'):
            return self.test_sample_list.shape[0]

    def create_file_list(self):
        sublist = []
        for i in range(len(self.filename)):
            sublist.append(
                self.filename[i].split('/')[-1].split('_')[2] + '_' + self.filename[i].split('/')[-1].split('_')[3]+ '_' + self.filename[i].split('/')[-1].split('_')[4])

        sublist_quchong = list(set(sublist))
        sublist_quchong.sort()
        sub_num = len(sublist_quchong)
        for i in range(int(len(sublist_quchong))):
            tmp_list = []
            tmp_list.append(sublist_quchong[i])
            self.test_sub.append(tmp_list)
            train_val = list(set(sublist_quchong) - set(tmp_list))
            random.shuffle(train_val)
            tmp = []
            num_val = 0
            for j in range(len(train_val)):
                if tmp_list[0].split('_')[0] == train_val[j].split('_')[0] and num_val < 5:
                    tmp.append(train_val[j])
                    num_val = num_val + 1
            tmp_train = list(set(train_val) - set(tmp))
            self.val_sub.append(tmp)
            self.train_sub.append(tmp_train)
        del sublist, sublist_quchong, tmp_list, tmp, num_val, tmp_train

    def read_sample_label_list(self):
        filename = self.filename #only the first frame of sample list
        fold = self.testsub
        train_sample_list = []
        train_label_list = []
        val_sample_list = []
        val_label_list = []
        test_sample_list = []
        test_label_list = []
        test_sample_name_list = []

        for i in range(len(filename)):
            label = int(filename[i].split('/')[-2])
            tmp = filename[i].split('/')[-1].split('_')[2] + '_' + filename[i].split('/')[-1].split('_')[3] + '_' + filename[i].split('/')[-1].split('_')[4]
            for j in range(len(self.train_sub[fold])):
                if tmp == self.train_sub[fold][j]:
                    train_sample_list.append(filename[i])
                    train_label_list.append(label)
            for k in range(len(self.val_sub[fold])) :
                if tmp == self.val_sub[fold][k]:

                    val_sample_list.append(filename[i])
                    val_label_list.append(label)

            for m in range(len(self.test_sub[fold])):
                if tmp == self.test_sub[fold][m]:
                    test_sample_list.append(filename[i])
                    test_label_list.append(label)
                    test_sample_name_list.append(filename[i].split('/')[-1])
        train_sample_list = np.array(train_sample_list)
        train_label_list = np.array(train_label_list)
        val_sample_list = np.array(val_sample_list)
        val_label_list = np.array(val_label_list)
        test_sample_list = np.array(test_sample_list)
        test_label_list = np.array(test_label_list)
        test_sample_name_list = np.array(test_sample_name_list)
        del tmp,  label

        return train_sample_list, train_label_list, \
               val_sample_list, val_label_list, \
               test_sample_list, test_label_list, test_sample_name_list


    def read_flow_data(self, sample_path):
        WIDTH = self.image_width
       
        x_img_path = sample_path
        # read flow_x
        img_x = Image.open(x_img_path).convert('RGB')
        img_x = img_x.resize((WIDTH, WIDTH), Image.NEAREST)
        img_x  = transform1(img_x)

        # read flow_y
        y_img_path = str.replace(sample_path, 'flow_x', 'flow_y')
        img_y = Image.open(y_img_path).convert('RGB')
        img_y = img_y.resize((WIDTH, WIDTH), Image.NEAREST)
        img_y= transform1(img_y)
   
        z_img_path = str(y_img_path).split('flow_y_')[1].strip('.jpg')
        cls = str(y_img_path).split('/')[2]


        frames_path = glob.glob('./data/preprocessing_o2a/cropped/'+cls+'/'+z_img_path+'/*')
        frames_path = [frame_path.replace("\\", "/") for frame_path in frames_path]
        for i,frame_path in enumerate(frames_path):
            #if i!=0 and i!=14:
                #continue
            frame = Image.open(frame_path).convert('RGB')
            frame = frame.resize((28,28),Image.NEAREST)
            frame = transform1(frame)
            frame=frame[None,:,:,:]
            #           D,C,H,W    通道数；深度frame_size，H,W对应的就是输入图像的高和宽
            if i == 0 :
                img_z = frame
            else:
                img_z=torch.cat((img_z, frame), dim = 0)

        img_x = img_x.unsqueeze(0)
        img_y = img_y.unsqueeze(0)
        imgs_tensor = torch.cat((img_x, img_y), 0)

        imgs_tensor = torch.cat((imgs_tensor, img_z), 0)
        del img_y, img_x, x_img_path, y_img_path, img_z   
        return imgs_tensor

