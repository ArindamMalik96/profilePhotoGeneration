import sys, os
import numpy as np
import tensorflow as tf
import cv2
import math
import shutil
import json
import pickle
import config_live as config
from tensorflow import keras
import smtplib
from email.message import EmailMessage
from datetime import datetime
import time
import sys
#import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
from math import floor
import json
import requests
from dist_bw_faces_multi1 import *
#import dlib
#from imutils import face_utils
#import imutils






def detect_faces (img):
    (h, w) = img.shape [:2]
    blob = cv2.dnn.blobFromImage (cv2.resize (img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    detections = face_model.forward()
    x, y, z, k = (0, 0, 0, 0)
    faces = []


    for i in range (0, detections.shape [2]):
      box = detections [0, 0, i, 3:7] * np.array ( [w, h, w, h])
      (startX, startY, endX, endY) = box.astype ("int")

      confidence = detections [0, 0, i, 2]

      # If confidence > 0.5, show box around face
      if (confidence > 0.5 and faces == []):
        x, y, z, k = (startX, startY, endX, endY)
        faces.append ([startX, startY, endY - startY, endX - startX])

      elif (confidence > 0.5):
        if ((startX > x and startX < z) and (startY > y and startY < k)):
          continue

        elif ((startX < x and endX > x) and (startY < y and endY > y)):
          area1 = (endX - x) * (endY - y)
          if (area1 / ((z - x) * (k - y))) > 0.1:
            faces.pop (-1)
          x, y, z, k = (startX, startY, endX, endY)
          faces.append ([startX, startY, endY - startY, endX - startX])
          continue

        else:
          if (endX < w and endY < h):
            x, y, z, k = (startX, startY, endX, endY)
            faces.append ([startX, startY, endY - startY, endX - startX])

    return faces

def download_image(url):        #needed
    req = urllib.request.Request(url, data=None, headers={
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
})

    #filename_w_ext = os.path.basename(url)
    #filename, file_extension = os.path.splitext(filename_w_ext)

    filename_w_ext = url[url.rfind("/")+1:]
    ind=filename_w_ext.index('.')
    filename=filename_w_ext[:ind]
    file_extension=filename_w_ext[ind:]

    #print("\nfilename:",filename_w_ext,"##",filename,"##",file_extension,"\n")
    img_file = urllib.request.urlopen(req)

    if url.endswith("png"):
        with open(filename_w_ext,'wb') as output:
            output.write(img_file.read())
        convert_png_to_jpeg(filename_w_ext)
    else:
        with open(filename_w_ext,'wb') as output:
            output.write(img_file.read())
    return filename_w_ext

def convert_png_to_jpeg(path_to_file):      
    im = Image.open(path_to_file)
    rgb_im = im.convert('RGB')
    
    name_of_file = os.path.basename(path_to_file)
    outfilename = name_of_file.split(".")[0] + ".jpeg"
    directory = os.path.dirname(path_to_file)
    outfile = os.path.join(directory, outfilename)
    print("Filname after conversion is", outfile)
    rgb_im.save(outfile)
    return outfile

def load_image (img_path):
    pic = cv2.imread (img_path)
    return pic

face_model = cv2.dnn.readNetFromCaffe (config.prototxt_path, config.caffemodel_path)


def sizeLimit (img, face):
    Y, X, Z = img.shape
    x, y, h, w = face[0], face[1], face[2], face[3]
    if (x < 0): x = 0
    if (y < 0): y = 0
    if ((y + h) > Y) : h = Y - y - 1
    if ((y + h) > Y) : w = X - x - 1

    a, b, c = img.shape
    if a == 0 or b == 0 or c == 0:
        return 0

    sizeL = [x, y , x + w, y + h]
    return sizeL

def getFaceFromCaffe (url):
    try:
        img_path123 =download_image (url)
        pic = load_image (img_path123)
        ret = getFaces(pic,img_path123)
        #os.remove(img_path123)
        return ret
    except Exception as e:
        return ''

def getPerfectProfilePhoto (url):
    try:
        img_path123 =download_image (url)
        pic = load_image (img_path123)
        ret = getFaces(pic,img_path123,True)
        #os.remove(img_path123)
        return ret
    except Exception as e:
        return ''


def getFaces(pic,img_path123,returnPath = False):
    faces = detect_faces (pic)
    num_faces = len (faces)
    Dimension_Faces=[None]*((num_faces))   
    faceSize=[0]*(num_faces)
    Y, X, Z = pic.shape

    if num_faces==0:
    	return ''
    m=[]
    maxW=0
    for i in range(num_faces):    
        x=[]
        x = sizeLimit(pic,faces[i])
        
        #margin
        diag = ((x[0] - x[2])**2 + (x[1] - x[3])**2)**(0.5)
        diag *= 0.45
        z=[]
        z.append((0 if (x[0]-int(diag/2)) <0 else (x[0]-int(diag/2))))  
        z.append((0 if (x[1]-int(diag/2)) <0 else (x[1]-int(diag/2))))
        z.append(((X-1) if (x[2]+int(diag/2)) >X else (x[2]+int(diag/2)))) 
        z.append((Y-1 if (x[3]+int(diag/2)) >Y else (x[3]+int(diag/2))))
        #drawBox(img_path123,z)

        #format
        width=str(z[2]-z[0])+'x'
        height=str(z[3]-z[1])+'+'
        x1=str(z[0])+'+'
        y1=str(z[1])
        Dimension_Faces[i]=width+height+x1+y1
        
        if maxW<((z[2]-z[0])*(z[3]-z[1])):
        	maxW=(z[2]-z[0])*(z[3]-z[1])
        	m=z
        
        faceSize[i]=(z[2]-z[0])*(z[3]-z[1])

    if(num_faces>1): 
        max_value_face = max(faceSize)
        max_value_face_index=faceSize.index(max_value_face)
        del faceSize[max_value_face_index]
        max_value_face2 = max(faceSize)
        proportion=max_value_face/max_value_face2
        if(proportion<1.3):
            print("\n\nMULTI USER, EXITING\n\n")
            return ""
        drawBox(img_path123,m)
        
        maxDim=Dimension_Faces[max_value_face_index]
    else:
        #drawBox(img_path123,m)
        maxDim=Dimension_Faces[0]
    
    if returnPath:
        im = cv2.imread(img_path123)
        im = cv2.rectangle(im, (m[0],m[1]), (m[2],m[3]), (0,255,255), 2) 
        im = im[m[1]:m[3],m[0]:m[2]]
        cv2.imwrite((img_path123), im)
        return img_path123

    return maxDim

def drawBox(path,z):
	image = cv2.imread(path) 
	# Window name in which image is displayed 
	window_name = 'Image'
  
# Start coordinate, here (5, 5) 
# represents the top left corner of rectangle 
	start_point = (z[0],z[1]) 
	end_point = (z[2],z[3]) 
  
	# Blue color in BGR 
	color = (255, 0, 0) 

	# Line thickness of 2 px 
	thickness = 2

	# Using cv2.rectangle() method 
	# Draw a rectangle with blue line borders of thickness of 2 px 
	image = cv2.rectangle(image, start_point, end_point, color, thickness) 
	cv2.imwrite((path), image)
	# Displaying the image  
	#cv2.imshow(window_name, image)  
	#sys.exit(0)