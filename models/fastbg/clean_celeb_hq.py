import cv2
import numpy as np
import glob
import os
import shutil

data_set = '/Users/agunin/Downloads/CelebAMask-HQ/results'
new_st = '/Users/agunin/Downloads/CelebAMask-HQ/new'

shutil.rmtree(new_st,ignore_errors=True)
os.makedirs(new_st,exist_ok=True)
os.mkdir(os.path.join(new_st,'images'))
os.mkdir(os.path.join(new_st, 'masks'))

files = glob.glob(data_set + '/masks/*.*')
for i in range(len(files)):
    mask = files[i]
    name = os.path.basename(mask)
    img = data_set + '/images/' + name
    files[i] = (name,img, mask)

count = 0
for name,img_file,mask_file in files:
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    maskq = np.copy(mask)
    maskq[maskq>0] = 1
    area = np.sum(maskq)
    w = maskq.shape[1]
    h = maskq.shape[1]
    area = np.sum(maskq)
    if area>w*h*0.7:
        continue
    area = np.sum(maskq[:,0:2])
    if area>h*2*0.1:
        continue
    area = np.sum(maskq[:, w-2:])
    if area > h * 2 * 0.1:
        continue
    area = np.sum(maskq[0:2,:])
    if area > w * 2 * 0.1:
        continue
    img = cv2.imread(img_file)
    img1 = img.astype(np.float32)
    mask = mask.astype(np.float32)/255
    mask = np.expand_dims(mask,axis=2)
    img1 = img1*mask
    img1 = img1.astype(np.uint8)
    res = np.concatenate([img,img1],axis=1)
    count+=1
    shutil.copy(mask_file,os.path.join(new_st,'masks',name))
    shutil.copy(img_file, os.path.join(new_st, 'images', name))

print(count)