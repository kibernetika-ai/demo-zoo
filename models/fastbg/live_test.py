from ml_serving.drivers import driver
import cv2
import os
import numpy as np


drv = driver.load_driver("model")()

drv.load_model('kuberlab-demo/person-mask:1.82.85')

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    if frame is None:
       break
    serv_img = cv2.resize(frame[:,:,::-1],(160,160))
    serv_img = serv_img.astype(np.float32)/255
    result = drv.predict({'image': np.expand_dims(serv_img, axis=0)})
    mask = result['output']
    mask = mask[0]*255
    #mask[mask < 10] = 0
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = mask.astype(np.float32)/255
    frame = frame.astype(np.float32)*np.expand_dims(mask,axis=2)
    frame = frame.astype(np.uint8)
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1)
    if key in [ord('q'), 202, 27]:
        break

video.release()
