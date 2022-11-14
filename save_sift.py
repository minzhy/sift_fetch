import cv2 as cv
import numpy as np
import sys
import time
import os

def float2(f):
    res = format(f,'.2f')
    return res

def float3(f):
    res = format(f,".3f")
    return res

if __name__ == '__main__':
    print(len(sys.argv))
    print(sys.argv[1])
    img_path = sys.argv[1]
    output_name = sys.argv[2]

    img = cv.imread(img_path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #--- detect keypoints using sift , compute descriptors
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)

    # tips: type of sortDes is list, but the type of des is ndarray so we change the operation in the following:
    lenKp = len(kp)
    index = [x for _,x in sorted(zip(kp,range(lenKp)),key=lambda x:x[0].size,reverse=True)]
    sortDes = des[index]
    sortKp = sorted(kp,key=lambda x : x.size, reverse=True)


    # --output
    with open(output_name,"a+") as f:
        f.seek(0)
        f.truncate()
        f.write(str(lenKp)+" 128\n")
        for i in range(lenKp):
            f.write(str(float2(sortKp[i].pt[1]))+" "+str(float2(sortKp[i].pt[0]))+" "+str(float2(sortKp[i].size))+" "+str(float3(sortKp[i].angle))+"\n") 
            # +" "+str(sortKp[i].pt[0])+" "+str(sortKp[i].size+" "+str(sortKp[i].angle)))
            for j in range(1,129):
                f.write(" "+str(int(sortDes[i][j-1])))
                if j % 20 == 0:
                    f.write("\n")
            f.write("\n")
            pass

    
    # img1 = cv.imread()
