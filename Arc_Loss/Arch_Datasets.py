import torch
import numpy as np
import torch.utils.data as data
import os
from PIL import Image,ImageDraw

class Datasets(data.Dataset):
    def __init__(self,img_path):
        super().__init__()
        self.img_path=img_path
        self.img_list=os.listdir(img_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name=self.img_list[index]
        lable=torch.Tensor([int(img_name[0])]).long()
        img_open=Image.open(os.path.join(self.img_path,img_name))
        #img_open.show()
        img_resize=img_open.resize((200,200))
        img_data=torch.Tensor(np.array(img_resize)/255).permute(2,0,1)
        return img_data,lable

if __name__ == '__main__':
    img_path=r"F:\Datasets_Test\Center_Classify\face_img"
    datasets=Datasets(img_path)
    data = datasets[9][0]
    lable=datasets[9][1]
    print(lable,data.shape)
