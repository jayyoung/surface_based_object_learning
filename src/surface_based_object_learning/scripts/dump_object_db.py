#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from mongodb_store.message_store import MessageStoreProxy
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
from soma_llsd_msgs.msg import *
from soma_msgs.msg import SOMAObject
from soma_manager.srv import *
from soma_llsd.srv import *
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
import cv
import cv2
import os
import pickle
import python_pcd
import sys
from object_interestingness_estimator.srv import *
import caffe
import scipy.misc
from PIL import Image
import numpy as np
import pandas as pd
import cPickle
import logging
import base64
import datetime, time
import sys
import numpy as np
import cv2


"""
This script handles the skimage exif problem.
"""

ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}


def open_oriented_im(im_path):
    im = Image.open(im_path)
    if hasattr(im, '_getexif'):
        exif = im._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            im = apply_orientation(im, orientation)
    img = np.asarray(im).astype(np.float32) / 255.
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def apply_orientation(im, orientation):
    if orientation in ORIENTATIONS:
        for method in ORIENTATIONS[orientation]:
            im = im.transpose(method)
    return im

class CNNWrapper():
    REPO_DIRNAME = os.path.abspath('/home/jxy/aloof/vision_stuff/caffe')
    default_args = {
        'model_def_file': (
            '{}/models/resnet/ResNet-152-deploy.prototxt'.format(REPO_DIRNAME)), #'{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/resnet/ResNet-152-model.caffemodel'.format(REPO_DIRNAME)), #'{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/models/resnet/mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        #print("MEAN: " + str(np.load(mean_file).mean(1).mean(1)))
        #sys.exit()
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify(self, image):
        try:
            print("starting")
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            #print(scores)
            endtime = time.time()
            print("prediction done")

            #indices = (-scores).argsort()[:15]
            indices = (-scores).argsort()
            predictions = self.labels[indices]
            print("woo here we go")
            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            print("done meta")
            bet_result = []
            #if(False):
            logging.info('result: %s', str(meta))
            print("calculating gain")
            # Compute expected information gain

            #expected_infogain = np.dot(self.bet['probmat'], scores[self.bet['idmapping']])
            #expected_infogain *= self.bet['infogain']

            print("done, sorting scores")
            # sort the scores
            #infogain_sort = expected_infogain.argsort()[::-1]
            #bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
            #              for v in infogain_sort[:10]]
            #bet_result = []
            #logging.info('bet result: %s', str(bet_result))
            print("done")

            return (True, meta, bet_result, '%.3f' % (endtime - starttime))

        except Exception as err:
            print(err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')




if __name__ == '__main__':
    rospy.init_node('om_test', anonymous = False)

    bridge = CvBridge()
    print("beginning")
    soma_query_service = rospy.ServiceProxy('/soma/query_objects',SOMAQueryObjs)
    segment_query_service = rospy.ServiceProxy('/soma_llsd/get_segment',GetSegment)
    scene_query_service = rospy.ServiceProxy('/soma_llsd/get_scene',GetScene)
    query = SOMAQueryObjsRequest()
    query.query_type = 0
    query.objecttypes=['unknown']
    response = soma_query_service(query)
    rospy.wait_for_service('/object_interestingness_estimator/estimate',10)
    interest_srv = rospy.ServiceProxy('/object_interestingness_estimator/estimate',EstimateInterest)
    bridge = CvBridge()
    #CNNWrapper.default_args.update({'gpu_mode': True})
    #c = CNNWrapper(**CNNWrapper.default_args)
    sift = cv2.SURF(1000)
    print("processing objects")
    for k in response.objects:
        print("getting: " + k.id)
        object_target_dir = "object_dump/"+str(eval(k.metadata)['waypoint'])+"/"+k.id+"/"
        #print("writing " + object_target_dir)
        segment_req = segment_query_service(k.id)
        print("getting seg imgs")

        for obs in segment_req.response.observations:
            #scene = scene_query_service(obs.scene_id)
            #scene_rgb = bridge.imgmsg_to_cv2(scene.response.rgb_img)
            rgb = obs.rgb_cropped
            cv_rgb_image = bridge.imgmsg_to_cv2(rgb)
            #cv_mask_image = bridge.imgmsg_to_cv2(obs.image_mask)
            #height, width, depth = cv_mask_image.shape
            #cv_mask_image = cv2.cvtColor(cv_mask_image,cv2.COLOR_RGB2GRAY)
            #_,thresh = cv2.threshold(cv_mask_image,1,255,cv2.THRESH_BINARY)
            #cv_mask_image = cv2.convertScaleAbs(cv_mask_image)
            #print(cv_mask_image.shape)
            ##print(cv_rgb_image.shape)
            #res = cv2.bitwise_and(scene_rgb,scene_rgb,mask = cv_mask_image)
            #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            #cnt = contours[0]
            #x,y,w,h = cv2.boundingRect(cnt)
            #crop = res[y:y+h,x:x+w]
            kp, des = sift.detectAndCompute(cv_rgb_image,None)
            print("kp:" + str(len(kp)))
            if(len(kp) < 20):
                continue
            #interest_points = interest_srv(obs.map_cloud)
            #print("des:" + str(len(des)))
            #if(interest_points.output.data >= 4):
            if not os.path.exists(object_target_dir):
                os.makedirs(object_target_dir)
            print("--- WRITING ---")
            print("accepting")
            cv2.imwrite(object_target_dir+str(segment_req.response.observations.index(obs))+"-"+str(len(kp))+".png",cv_rgb_image)



    print("all done!")
