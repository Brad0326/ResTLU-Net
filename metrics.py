import SimpleITK as sitk
import os
import numpy as np
import time
# 实验用的图片大小为320*512*512，
# 即矢状面（x轴方向）切片数为320，冠状面（y轴方向）切片数为512，
# 横断面（z轴方向）片数为512
start=time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# FN=0   #FN表示在在手工标签中是脑组织，在生成标签中是非脑组织
# FP=0   #FP表示在生工标签中是非脑组织，在生成标签中是脑组织
# TP=0   #TP表示在手工标签中是脑组织，在生成标签中也是脑组织
# TN=0   #TN表示在手工标签中为非脑组织，在生成标签中也是非脑组织
# UO=0    在手工标签或者生成标签中是脑组织

Dice=[]
Sen=[]
Spc=[]
IoU=[]
Jaccard=[]
VOE=[] #体素重叠误差 Volumetric Overlap Error
FNR=[] #欠分割率 False negative rate
FPR=[] #过分割率 False positive rate
PPV=[]
#依次遍历mask图像像素点
def compare(data1,data2):
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    UO=0
    for i in range(0,data1.shape[0]):
        for j in range(0,data1.shape[1]):
            for k in range(0,data1.shape[2]):
                if data1[i][j][k]==1 and data2[i][j][k]==0:
                    FN=FN+1
                if data1[i][j][k]==0 and data2[i][j][k]==1:
                    FP=FP+1
                if data1[i][j][k] ==1 and data2[i][j][k] == 1:
                    TP=TP+1
                if data1[i][j][k] == 0 and data2[i][j][k] == 0:
                    TN=TN+1
                if data1[i][j][k] == 1 or data2[i][j][k] == 1:
                    UO=UO+1
    dice=2*TP/(2*TP+FP+FN)
    sen=TP/(TP+FN)
    spc=TN/(TN+FP)
    iou=TP/UO
    jac=TP/(FN+FP+TP)
    voe=1-TP/(FN+FP+TP)
    fnr=FN/(FN+FP+TP)
    fpr=FP/(FN+FP+TP)
    ppv=TP/(TP+FP)
    print("戴斯系数为：",dice)
    print("灵敏度为：",sen)
    print("特异度为:",spc)
    print("交并比为:",iou)
    print("杰卡德系数：",jac)
    print("体素重叠误差:",voe)
    print("欠分割率:",fnr)
    print("过分割率",fpr)
    print("PPV",ppv)
    Dice.append(dice)
    #print(result)
    Sen.append(sen)
    #print(result)
    Spc.append(spc)
    #print(result)
    IoU.append(iou)
    Jaccard.append(jac)
    VOE.append(voe)
    FNR.append(fnr)
    FPR.append(fpr)
    PPV.append(ppv)


def get_data(path1,path2):
    length1=len([lists for lists in os.listdir(path1) if os.path.isfile(os.path.join(path1, lists))])
    length2=len([lists for lists in os.listdir(path2) if os.path.isfile(os.path.join(path2, lists))])
    if length1!=length2:
        print("文件数目错误")
        return 0
    else:
        dir1 = path1
        list1 = os.listdir(dir1)
        list1 = sorted(list1)
        dir2 = path2
        list2 = os.listdir(dir2)
        list2 = sorted(list2)
        for i in range(0, length1):
            # data1 = os.path.join(dir1, list1[i])
            # print(data1)
            # pre_data1 = data1
            pre_data1 = os.path.join(dir1, list1[i])
            print(list1[i])
            #path1 = './data_analysis/subject-11-label.nii'
            image1 = sitk.ReadImage(pre_data1)
            shape_img1 = image1.GetSize()
            print(f'shape of image1: {shape_img1}')

            # convert to ndarry
            data1 = sitk.GetArrayFromImage(image1)
            shape_data1 = data1.shape
            print(f'shape of data1: {shape_data1}')

            # data2 = os.path.join(dir2, list2[i])
            # print(data2)
            # pre_data2 = data2
            pre_data2 = os.path.join(dir2, list2[i])
            print(list2[i])
            #path2 = './data_test/subject-11-label.nii'
            image2 = sitk.ReadImage(pre_data2)
            shape_img2 = image2.GetSize()
            print(f'shape of image2: {shape_img2}')

            # convert to ndarry
            data2 = sitk.GetArrayFromImage(image2)
            shape_data2 = data2.shape
            print(f'shape of data2: {shape_data2}')
            compare(data1,data2)


#path1：手工标签路径  path2:生成标签路径
get_data('/home/user/Desktop/data/test_data_8_mask','/home/user/Desktop/feihong_416/shujufenxi/tool_mask/AFNI')
#print(result)
for i in range(0,len(Dice)):
    print('Dice:  | sen:   | spc:   |iou:    |jac:    |voe:    |fnr:     fpr|ppv:     ppv')
    print('%2.4f | %2.4f | %2.4f | %2.4f  |%2.4f | %2.4f | %2.4f | %2.4f' %
          (Dice[i], Sen[i], Spc[i], IoU[i], Jaccard[i], VOE[i], FNR[i], FPR[i],PPV[i]))
    #print(result)
print(Dice)
print(Sen)
print(Spc)
print(IoU)
print(Jaccard)
print(VOE)
print(FNR)
print(FPR)
print(PPV)
end=time.time()
print('Running time:%s minutes'%((end-start)/60))