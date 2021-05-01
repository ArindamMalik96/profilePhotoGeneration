"""Utilities for cropping faces, marking landmarks and finding distance between two images after cropping faces.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import align.detect_face
import sys
from scipy import misc
from scipy.ndimage.interpolation import rotate
from math import atan, degrees, radians, pi, sin
#import matplotlib.path as mplPath
#import csv

from PIL import Image, ImageDraw, ExifTags, ImageFont
import urllib.request

def get_image(image_path):      #needed
    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
        return np.zeros((160,160,3))
    else:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            return np.zeros((160,160,3))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]
    return img

def rotate_img(img, angle):
    return rotate(img,angle)

def get_cropped_face(img, margin, crop_size, i):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    
    print("Bounding Boxes", bounding_boxes)
    print("Points", points)
    
    nrof_faces = bounding_boxes.shape[0]
    det = bounding_boxes[:,0:4]
    img_size = np.asarray(img.shape)[0:2]
    index = 0
    if nrof_faces == 0:
        print("Could not crop image #"+str(i))
        return img[0:crop_size,0:crop_size,:]
    if nrof_faces > 1:
        print("Found multiple faces, find most centered or large face")
        bounding_box_size = np.min([det[:,2]-det[:,0],det[:,3]-det[:,1]])
#        img_center = img_size / 2
#        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
#        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        index = np.argmax(bounding_box_size)#-offset_dist_squared*2.0) # some extra weight on the centering
        det = np.squeeze(det[index,:])
    else:
        det = np.squeeze(det[0,:])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    eye1 = [points[0][index], points[5][index]]
    eye2 = [points[1][index], points[6][index]]
    diff = np.asarray(eye2)-np.asarray(eye1)
    rotate_angle = degrees(2*pi-atan(-diff[1]/diff[0]))
    print(rotate_angle)
    if abs(sin(radians(rotate_angle))) > 0.05:
        return get_cropped_face(rotate_img(img,rotate_angle), margin, crop_size, i)
    
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    scaled = misc.imresize(cropped, (crop_size, crop_size), interp='bilinear')
    misc.imsave("cropped_image"+str(i)+".png", scaled)
    scaled = facenet.prewhiten(scaled)    
    return scaled

def mark_faces_in_a_pic(imgPath):       #needed
    try:
        image=Image.open(imgPath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())
    
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        image.save(imgPath)
        image.close()

    except (AttributeError, KeyError, IndexError):
    # cases: image don't have getexif
        pass
    minsize = 70
    margin = 320
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor    
    img = get_image(imgPath)
    source_img = Image.open(imgPath).convert("RGBA")
    with tf.Graph().as_default():
        with tf.Session() as sess:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    
#    print("Bounding Boxes", bounding_boxes)
#    print("Points", points)
    
    nrof_faces = bounding_boxes.shape[0]
    det = bounding_boxes[:,0:4]
    img_size = np.asarray(img.shape)[0:2]
    
    if nrof_faces == 0:
        #print("Could not find faces")
        a=""
        return a,0
    if nrof_faces > 1:
        a=""
        #print("Found", nrof_faces,"faces")
    else:
        z=1
         #print("Found a single face")
        
    # make a blank image for the rectangle, initialized to a completely transparent color
    #tmp = Image.new('RGBA', source_img.size, (0,0,0,0))
    Dimension_Faces=[None]*((nrof_faces))   
    # get a drawing context for it
    #draw = ImageDraw.Draw(tmp)
    faceSize=[0]*(nrof_faces)

    for i in range(nrof_faces):    
        det1 = np.squeeze(det[i,:])
        bb = np.zeros(4, dtype=np.int32)
        diag=(((det1[2]-det1[0])*(det1[2]-det1[0]))+((det1[3]-det1[1])*(det1[3]-det1[1])))**(0.5)
        margin=diag*(0.45)
        bb[0] = np.maximum(det1[0]-margin/2, 0)
        bb[1] = np.maximum(det1[1]-margin/2, 0)
        bb[2] = np.minimum(det1[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det1[3]+margin/2, img_size[0])
        ww=(bb[2]-bb[0])*(bb[3]-bb[1])
        width=str(bb[2]-bb[0])+'x'
        height=str(bb[3]-bb[1])+'+'
        x1=str(bb[0])+'+'
        y1=str(bb[1])
        Dimension_Faces[i]=width+height+x1+y1
        faceSize[i]=(bb[2]-bb[0])*(bb[3]-bb[1])
        #draw.rectangle(((bb[0], bb[1]), (bb[2], bb[3])), fill=(0,0,0,0), outline="green")
        #fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        #for j in range(5):
        #  draw.text((points[j][i],points[5+j][i]), str(j), fill=(255,255,255,255))
#            draw.ellipse((points[j][i]-2, points[5+j][i]-2, points[j][i]+2, points[5+j][i]+2), fill="red")
    
    ###  
    if(nrof_faces>1): 
        max_value_face = max(faceSize)
        max_value_face_index=faceSize.index(max_value_face)
    #    print("\n\n",faceSize,"@@@@@@@@",max_value_face,"@@@@@@@@",max_value_face_index)
        del faceSize[max_value_face_index]
        max_value_face2 = max(faceSize)
        proportion=max_value_face/max_value_face2
    #    print("\n\n",faceSize,"@@@@@@@@",max_value_face2,"@@@@@@@",proportion,"\n\n")
        if(proportion<1.3):
            print("\n\nMULTI USER, EXITING\n\n")
            return "",0
        
        maxDim=Dimension_Faces[max_value_face_index]
    else:
        maxDim=Dimension_Faces[0]    
        #print(maxDim)
    
    #for i in range(nrof_faces):
    #    for j in range(5):
    #        print("point #", j, ": ",points[j][i],points[5+j][i])

    #x = (points[1][i] - points[0][i])*0.15
    #points1 = ((points[0][i]+x, points[5][i]),(points[1][i]-x, points[6][i]),(points[4][i]-x, points[9][i]),(points[3][i]+x, points[8][i]))
    #draw.polygon((points1))

    #bbPath = mplPath.Path(np.array([[points[0][i]+x, points[5][i]],
    #                 [points[1][i]-x, points[6][i]],
    #                 [points[4][i]-x, points[9][i]],
    #                 [points[3][i]+x, points[8][i]]]))
    #print("is point inside?",bbPath.contains_point((points[2][i], points[7][i])));


    #source_img = Image.alpha_composite(source_img, tmp)
    #source_img.save("landmarks.png", "PNG")
    #print("\n\nPASSED\n\n")
    return maxDim, nrof_faces

def dist_between(image1_path, image2_path, sess=None, margin = 32, crop_size = 160):
    
    img1 = get_image(image1_path)
    img2 = get_image(image2_path)     
    
    cropped_img1 = get_cropped_face(img1, margin, crop_size, 1)
    cropped_img2 = get_cropped_face(img2, margin, crop_size, 2)
    
    pair = np.asarray([cropped_img1,cropped_img2])
   
    #print(pair.shape)
    
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    #print('Runnning forward pass on LFW images')
    emb_array = np.zeros((2, embedding_size))
    feed_dict = { images_placeholder:pair, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    #print(emb_array)
    diff = np.subtract(emb_array[[0],:], emb_array[[1],:])
    dist = np.sum(np.square(diff),1)
    return dist

def convert_png_to_jpeg(path_to_file):      #needed
    im = Image.open(path_to_file)
    rgb_im = im.convert('RGB')
    
    name_of_file = os.path.basename(path_to_file)
    outfilename = name_of_file.split(".")[0] + ".jpeg"
    directory = os.path.dirname(path_to_file)
    outfile = os.path.join(directory, outfilename)
    print("Filname after conversion is", outfile)
    rgb_im.save(outfile)
    return outfile

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

def checker():
    file1 = open("/home/arindam/Desktop/projects/FaceDetection/facedetection2018-07-19-03.csv", 'r')
    reader = csv.reader(file1)
    new_rows_list = []
    file2 = open("/home/arindam/Desktop/projects/FaceDetection/NEW---facedetection2018-07-19-03.csv", 'w')
    writer = csv.writer(file2)
    
    for row in reader:
            image_url=row[0]
            if '/var/www/html/web' in image_url:
                image_url='https://www.jeevansathi.com'+image_url[17:]
            print("~~~~~~~~~~~",image_url)
            downloaded_image=download_image(image_url)
            bb_facenet, nrof_faces=mark_faces_in_a_pic(downloaded_image)
            new_row = [row[0], row[1],nrof_faces,bb_facenet]
            
            writer.writerow(new_row)   
            new_rows_list.append(new_row)

    file1.close()   # <---IMPORTANT

#    writer.writerows(new_rows_list)
    file2.close()

def mark_download(args):     #needed
    
    #filename = input("Enter image Path\n")
    print("Mark::",args)
    Dimension_Faces,nrof_faces=    mark_faces_in_a_pic(download_image(args))
    print("DONE::",Dimension_Faces)
    return Dimension_Faces
#   print("\n\n%%%%%%%%%%%%%%%%",Dimension_Faces,"%%%%%%",nrof_faces,"%%%%%%%%%%%%")
#    get_cropped_face(get_image(filename), 32, 160, 1)
#    misc.imsave("test_rotate.jpg",rotate(get_image("test.jpg"),30))    
##    get_cropped_face(get_image("test.jpg"), 32, 160, 1)
#    with tf.Graph().as_default():
#        with tf.Session() as sess:
#            
#            # Load the model
#            facenet.load_model("model_checkpoints/20180402-114759/")
#            
#            for i in range(10):
#                print("Distance = ",dist_between(input("Enter image 1 Path\n"),input("Enter image 2 Path\n"), sess))
            
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
#    parser.add_argument('model', type=str, 
#        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')      
    #parser.add_argument('--lfw_batch_size', type=int,
    #    help='Number of images to process in a batch in the LFW test set.', default=100)
    #parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    #parser.add_argument('--crop_size', type=int,
    #    help='Size of the bounding box (height, width) in pixels.', default=160)
    #parser.add_argument('--filename', type=str,
    #    help='Filename of the input image whose face is to be cropped', default='')
    parser.add_argument('--url', type=str,
        help='URL of the input image whose face is to be cropped', default='')       
    return parser.parse_args(argv)

if __name__ == '__main__':
   mark_download(parse_arguments(sys.argv[1:]))
   #checker()
