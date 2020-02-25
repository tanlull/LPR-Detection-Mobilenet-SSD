import cv2 as cv
import cv2
import numpy as np 
import os
from align import *
import params 

#cvNet = cv2.dnn.readNetFromCaffe("./mssd512_voc.prototxt" , "./mssd512_voc.caffemodel" )
cvNet = cv2.dnn.readNetFromCaffe("./lpr.prototxt" , "./lpr.caffemodel" )

def detect(out_file,im):
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
    #print(im_tensor.shape)
    import time
    cvOut = cvNet.forward()
    for _ in range(1):
        t0 =time.time()
        cvOut = cvNet.forward()
        #print(time.time() -t0)
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
            #cv2.imshow("cropped" , cropped)

            #cv2.waitKey(0)
            cv2.rectangle(to_draw, (left,top) , (right,bottom) , (0,0,255) , 3)
                #Save file
            cropped_file_name = out_file+".cropped.jpeg" 
            cv2.imwrite(cropped_file_name, cropped) 
            cv2.imwrite(out_file, to_draw)
            print("Saved: "+out_file)
    return to_draw
    #cv2.imshow('image' , to_draw)
    #cv2.waitKey(0)
    


folder_out = "./output"

skipFrame = int(params.get("RTSP","skip"))# input number of frame to be skipped processing  
frameNo = 0

winName = 'LPR'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

# Get a reference to webcam #0 (the default one)
v_source = params.get("RTSP","source1")
video_capture = cv2.VideoCapture(v_source)
while True:
    if(frameNo%skipFrame == 0) : 
        # Grab a single frame of video
        ret, frame = video_capture.read()
            # Display the resulting image
        out_file = os.path.join(folder_out, "{}.jpeg".format(frameNo))
        frame_detect = detect(out_file,frame)
        cv2.imshow(winName, frame_detect)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frameNo +=1
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
