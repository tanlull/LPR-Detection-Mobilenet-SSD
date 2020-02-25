import cv2 as cv
import cv2
import numpy as np 
import os
from align import *

cvNet = cv2.dnn.readNetFromCaffe("./mssd512_voc.prototxt" , "./mssd512_voc.caffemodel" )

def detect(filename,im):
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    to_draw = im.copy()
    pixel_means=[0.406, 0.456, 0.485]
    pixel_stds=[0.225, 0.224, 0.229]
    pixel_scale=255.0
    rows,cols,c  = im.shape
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    im = im.astype(np.float32)
    for i in range(3):
      im_tensor[0, i, :, :] = (im[:, :, 2 - i]/pixel_scale - pixel_means[2 - i])/pixel_stds[2-i]
    cvNet.setInput(im_tensor)
    print(im_tensor.shape)
    import time
    cvOut = cvNet.forward()
    for _ in range(1):
        t0 =time.time()
        cvOut = cvNet.forward()
        print(time.time() -t0)
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        #print("score = "+str(score))
        if score > 0.6:
            print("score = "+str(score))
            left =int( detection[3] * cols)
            top =int( detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)
            cropped = to_draw[top:bottom, left:right]
            #cropped = align(cropped)
            cv2.imshow("cropped" , cropped)

            #cv2.waitKey(0)
            cv2.rectangle(to_draw, (left,top) , (right,bottom) , (0,255,0) , 1)
                #Save file
            save_file_name = filename+".detected.jpeg" 
            cv2.imwrite(save_file_name, to_draw) 
            print("Saved:"+save_file_name)  
    
    cv2.imshow('image' , to_draw)
    cv2.waitKey(0)
    
  

folder = "./input"
folder_out = "./output"
for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    out_path = os.path.join(folder_out, filename)
    if filename.lower().endswith(".jpeg"):
        print(filename)
        image = cv2.imread(path)
        #image  = align(image)
        detect(out_path,image)
        print("-----------")


