"""An example of how to use your own dataset to train a classifier that recognizes people.
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
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
from scipy.ndimage.interpolation import rotate
import align.detect_face
from PIL import Image, ExifTags
import urllib.request

from math import atan, degrees, radians, pi, sin
def get_image(image_path):
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
    return "cropped_image"+str(i)+".png"

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            for sub_dir in [x[0] for x in os.walk(args.data_dir)]:
                for file in os.listdir(sub_dir):
                    if not file.startswith("cropped_") and not sub_dir == args.data_dir:
                        print("removing ", os.path.join(str(sub_dir), file))
                        os.remove(os.path.join(str(sub_dir), file))
            
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
                elif (args.mode=='CLASSIFY_IMAGE'):
                    dataset = train_set                
            else:
                dataset = facenet.get_dataset(args.data_dir)
     
            label_names = []
            
            for img_class in dataset:
                label_names.append(img_class.name)
            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            print("Total number of batches: ",nrof_batches_per_epoch)
#            for i in range(nrof_batches_per_epoch):
#                start_index = i*args.batch_size
#                end_index = min((i+1)*args.batch_size, nrof_images)
#                paths_batch = paths[start_index:end_index]
#                images = facenet.load_data(paths_batch, False, False, args.image_size)
#                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
#                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
#                print("Done with batch", i)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                #model = SVC(kernel='linear', probability=True)
                model = SVC(C=1, kernel='rbf', probability=True, gamma=2)
#                with open("training_data.pkl",'wb') as train_file:
#                    pickle.dump([emb_array,labels],train_file)
                    
                with open("training_data.pkl",'rb') as train_file:
                    [emb_array,labels] = pickle.load(train_file)                
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                
                results = np.equal(best_class_indices, labels)
                for i in range(20):
                    print(100-i*5, "percentile in failed classes", np.percentile(best_class_probabilities[results==0],100-i*5))
                    print(i*5,"percentile in successful classes", np.percentile(best_class_probabilities[results==1],i*5))
                print('Accuracy: %.3f' % accuracy)
                
            elif (args.mode=='CLASSIFY_IMAGE'):
                with tf.Graph().as_default():
      
                    with tf.Session() as sess:
                        # Load the model
                        print('Loading feature extraction model')
                        facenet.load_model(args.model)
                        for i in range(15):
                            print(classify_image_by_URL(input("Enter URL of image to classify\n"), args, sess))

def convert_png_to_jpeg(path_to_file):
    im = Image.open(path_to_file)
    rgb_im = im.convert('RGB')
    
    name_of_file = os.path.basename(path_to_file)
    outfilename = name_of_file.split(".")[0] + ".jpg"
    directory = os.path.dirname(path_to_file)
    outfile = os.path.join(directory, outfilename)
    print("Filname after conversion is", outfile)
    rgb_im.save(outfile)
    return outfile

def download_image(url):
    req = urllib.request.Request(url, data=None, headers={
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
})
    img_file = urllib.request.urlopen(req)
    if url.endswith("png"):
        with open("downloaded.png",'wb') as output:
            output.write(img_file.read())
        convert_png_to_jpeg("downloaded.png")
    else:
        with open("downloaded.jpg",'wb') as output:
            output.write(img_file.read())
    return "downloaded.jpg"

def classify_image_by_URL(url, args, sess):
    return classify_image(download_image(url),args,sess)

def classify_image(filename, args, sess):
    return classify_cropped_image(get_cropped_face(get_image(filename),args.margin,args.image_size,1),args,sess)

def classify_cropped_image(filename, args, sess):     
            nrof_images = 1
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            paths_batch = [filename]
            images = facenet.load_data(paths_batch, False, False, args.image_size)
            feed_dict = {images_placeholder:images, phase_train_placeholder:False}
#            print(sess.run(embeddings, feed_dict=feed_dict))
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            print('Testing classifier')
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)            
            print('Loaded classifier model from file "%s"' % classifier_filename_exp)
            predictions = model.predict_proba(emb_array)
            ind = np.argmax(predictions, axis=1)
            print(predictions[0][ind[0]])
            return class_names[ind[0]]
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
#        paths = []
#        for path in cls.image_paths:
#            print(path)
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY','CLASSIFY_IMAGE'],
        help='Indicates if a new classifier should be trained or a classification or a classification of single image ' + 
        'model should be used for classification', default='TRAIN')
    parser.add_argument('--filename', type=str,
        help='Filename to classify in case of CLASSIFY_IMAGE ' + 
        'model should be used for classification', default='test.jpg')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.', default="downloads")
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='model_checkpoints/20180402-114759')
    parser.add_argument('--classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.', default="classifier_celebrity.pkl")
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', default="True", action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.', default='downloads')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=20)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
