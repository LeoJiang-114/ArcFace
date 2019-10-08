import numpy as np
import torch
import os
import PIL.Image as Image
import matplotlib.pyplot as plt

img_path=r"F:\Datasets_Test\Center_Classify\face_test"
net_path=r"F:\jkl\Arc_Loss\Net_Save\Face_Arc.pth"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

net=torch.load(net_path)
net=net.to(device)
net.eval()
plt.ion()
list=[]
for img_name in os.listdir(img_path):
       img_open=Image.open(os.path.join(img_path,img_name))
       #img_open = Image.open(r"F:\Test_File\Center_Classify\Writing0-9\3.jpg").convert("L")
       img_resize = img_open.resize((200,200))
       # img_resize.show()
       img_data = torch.Tensor(np.array(img_resize) / 255).permute(2,0,1).unsqueeze(0)
       #print(img_data)
       img_data = img_data.to(device)
       feature_, classify_ = net(img_data,1,1)
       feature, classify = feature_[0].cpu().data.numpy(), classify_[0].cpu().data

       np.set_printoptions(suppress=True)
       #print(feature.numpy(), classify.numpy())
       list.append(feature)
boxes=np.array(list)
print(boxes,)

test_img_path=r"F:\Datasets_Test\Center_Classify\me.jpg"
test_open=Image.open(test_img_path)
test_resize=test_open.resize((200,200))
test_data=torch.Tensor(np.array(test_resize)/255).permute(2,0,1).unsqueeze(0)
test_data=test_data.to(device)
t_f_,t_c_=net(test_data,1,1)
t_f,t_c=t_f_[0].cpu().data.numpy(),t_c_[0].cpu().data.numpy()
np.set_printoptions(suppress=True)
print("Test_point:",t_f,t_c)

dic={0:"YZS",1:"ZJC",2:"WQS",3:"ME",4:"ZM",5:"LL",6:"QSB",7:"TH",8:"WXX"}
#求余弦距离
cosa=np.true_divide(np.sum((boxes*t_f),axis=1),np.sqrt(np.sum(boxes**2,axis=1))*np.sqrt(np.sum(t_f**2)))
cosa=np.maximum(cosa,0)#负数取0

#print(np.sqrt(np.sum(boxes**2,axis=1)))
print("余弦相似度：",cosa)
cos_belong=np.argmax(cosa)
print("结果：",dic[int(cos_belong)])

color_dic = {0: 'aqua', 1: 'blue', 2: 'black', 3: 'red', 4: 'green',
             5: 'pink', 6: 'purple', 7: 'orange',8: 'yellow'}
print(int(cos_belong))
for i,box in enumerate(boxes):
       plt.scatter(box[0], box[1], s=100, c=color_dic[i])
plt.legend(["YZS","ZJC","WQS","ME","ZM","LL","QSB","TH","WXX"])
plt.scatter(t_f[0],t_f[1],s=200,c=color_dic[int(cos_belong)])
plt.show()
plt.pause(0)