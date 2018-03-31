import cv2
import numpy as np

#######   training part    ###############
samples = np.loadtxt('generalsamples01.data',np.float32)
responses = np.loadtxt('generalresponses01.data',np.float32)
responses = responses.reshape((responses.size,1))

# model = cv2.KNearest()
model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

# model.train(samples,responses)

############################# testing part  #########################

im = cv2.imread(r'C:\Users\PPC\git\openCVTutorial\images\dPaE822 - Copy.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

return_string =[]
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest (roismall, k = 1)
            my_string = str(int((results[0][0])))
            cv2.putText(out,my_string,(x,y+h),0,1,(0,255,0))
            return_string.append(my_string)
            # cv2.imshow('im', out)
            # cv2.waitKey(0)

# print return_string
print return_string[::-1]
cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)