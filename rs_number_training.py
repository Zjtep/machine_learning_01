import sys

import numpy as np
import cv2

im = cv2.imread(r'C:\Users\PPC\git\openCVTutorial\images\training_03.png')
im3 = im.copy()

# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(3,3),0)
# blur = cv2.medianBlur(gray, 3)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,3,2)

# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

cv2.imwrite("asdf.png", thresh)
cv2.imshow("Output", thresh)
cv2.waitKey(0)

# cv2.waitKey(0)

#################      Now finding Contours         ###################

_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]
keys.append(104)
for cnt in contours:

    # if cv2.contourArea(cnt)>50:
    if 500>cv2.contourArea(cnt) > 50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        # print cv2.contourArea(cnt)
        if  h>27:
            # print h
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                print chr(key)
                # responses.append(int(chr(key)))
                responses.append(chr(key))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('rs_generalsamples.data',samples)
np.savetxt('rs_generalresponses.data',responses)