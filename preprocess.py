#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import cv2
import csv
import time
import collections
import mxnet as mx
import json
from json import load
from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel

pred_type = collections.namedtuple('prediction', ['slice', 'close', 'color'])
pred_types = {'face': pred_type(slice(0, 17), False, (173.91, 198.9, 231.795, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), False, (255., 126.99,  14.025, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), False, (255., 126.99,  14.025, 0.4)),
              'nose': pred_type(slice(27, 31), False, (160,  60.945, 112.965, 0.4)),
              'nostril': pred_type(slice(31, 36), False, (160,  60.945, 112.965, 0.4)),
              'eye1': pred_type(slice(36, 42), True, (151.98, 223.125, 137.955, 0.3)),
              'eye2': pred_type(slice(42, 48), True, (151.98, 223.125, 137.955, 0.3)),
              'lips': pred_type(slice(48, 60), True, (151.98, 223.125, 137.955, 0.3)),
              'teeth': pred_type(slice(60, 68), True, (151.98, 223.125, 137.955, 0.4))}


class BaseAlignmentorModel:
    def __init__(self, prefix, epoch, shape, gpu=-1, verbose=False):
        self._device = gpu
        self._ctx = mx.cpu() if self._device < 0 else mx.gpu(self._device)

        self.model = self._load_model(prefix, epoch, shape)
        self.exec_group = self.model._exec_group

        self.input_shape = shape[-2:]
        self.pre_landmarks = None

    def _load_model(self, prefix, epoch, shape):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', shape)], for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    @staticmethod
    def draw_poly(src, landmarks, stroke=1, color=(125, 255, 125), copy=True):
        draw = src.copy() if copy else src

        for pred in pred_types.values():
            le = [landmarks[pred.slice].reshape(-1, 1, 2).astype(np.int32)]
            cv2.polylines(draw, le, pred.close, pred.color, thickness=stroke)

        return draw


class CoordinateAlignmentModel(BaseAlignmentorModel):
    def __init__(self, prefix, epoch, gpu=-1, verbose=False):
        shape = (1, 3, 192, 192)
        super().__init__(prefix, epoch, shape, gpu, verbose)
        self.trans_distance = self.input_shape[-1] >> 1
        self.marker_nums = 106
        self.eye_bound = ([35, 41, 40, 42, 39, 37, 33, 36],
                          [89, 95, 94, 96, 93, 91, 87, 90])

    def _preprocess(self, img, bbox):
        maximum_edge = max(bbox[2:4] - bbox[:2]) * 3.0
        scale = (self.trans_distance << 2) / maximum_edge
        center = (bbox[2:4] + bbox[:2]) / 2.0
        cx, cy = self.trans_distance - scale * center

        M = np.array([[scale, 0, cx], [0, scale, cy]])

        corpped = cv2.warpAffine(img, M, self.input_shape, borderValue=0.0)
        inp = corpped[..., ::-1].transpose(2, 0, 1)[None, ...]

        return mx.nd.array(inp), M

    def _inference(self, x):
        self.exec_group.data_arrays[0][0][1][:] = x.astype(np.float32)
        self.exec_group.execs[0].forward(is_train=False)
        return self.exec_group.execs[0].outputs[-1][-1]

    def _postprocess(self, out, M):
        iM = cv2.invertAffineTransform(M)
        col = np.ones((self.marker_nums, 1))

        out = out.reshape((self.marker_nums, 2))

        pred = out.asnumpy()
        pred += 1
        pred *= self.trans_distance

        # add a column
        # pred = np.c_[pred, np.ones((pred.shape[0], 1))]
        pred = np.concatenate((pred, col), axis=1)
        
        return pred @ iM.T  # dot product

    def _calibrate(self, pred, thd):
        if self.pre_landmarks is not None:
            for i in range(self.marker_nums):
                if sum(abs(self.pre_landmarks[i] - pred[i]) < thd) != 2:
                    self.pre_landmarks[i] = pred[i]
        else:
            self.pre_landmarks = pred

        return self.pre_landmarks

    def get_landmarks(self, image, detected_faces=None, calibrate=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

        Arguments:
            image {numpy.array} -- The input image.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None}, format: {x1, y1, x2, y2, score})
        """

        for bbox in detected_faces:
            inp, M = self._preprocess(image, bbox)
            out = self._inference(inp)
            pred = self._postprocess(out, M)

            yield self._calibrate(pred, .8) if calibrate else pred

if __name__ == '__main__':
    
    #person name and 68landmarks coord
    labels = []
    #title
    frame = []
    #xpts and ypts
    coord = []
    path = "../Gazetracker/gazecapture"
    #create .csv
    csvfile = open('new9landmarks.csv','w',newline='') 
    writer=csv.writer(csvfile,delimiter=',')
    frame.append('name')
    for j in range(1, 19):
        frame.append('x'+str(j))
        frame.append('y'+str(j))
    frame.append('gtx')
    frame.append('gty')
    writer.writerow(frame)
    fd = MxnetDetectionModel("weights/16and32", 0, scale=.4, gpu=-1)
    fa = CoordinateAlignmentModel('weights/2d106det', 0)
    i = 0
    num1=0
    for cat in os.listdir(path):
        cat_path=os.path.join(path, cat)
        coord = []
        i += 1
        for image in os.listdir(cat_path):
            img_path = os.path.join(cat_path, image)
            if image == "dotInfo.json":
                jsonFile = open(img_path,'r')
                a = json.load(jsonFile)
                for k in a:
                    if k=="XPts" or k=="YPts" :
                        length=len(a[k])
                        for j in range(length):    
                            coord.append(a[k][j])               
            if image  == "frames":
                num1 = 0
                total = 0
                
                for image_ in sorted(os.listdir(img_path)):
                    print("img_path: ", img_path)
                    print("loaded image", image_)
                    labels = []
                    labels.append(i)
                    img=cv2.imread(img_path+'/'+image_)
                    # face detector
                    bboxes = fd.detect(img)
                    # face landmarks coordinate (68 point)
                    for pred in fa.get_landmarks(img, bboxes, True):
                        num = 0
                        for p in np.round(pred).astype(int):
                           num+=1
                           #依序是左眼左邊左眼瞳孔左眼右邊
                           if num==36 or num==39 or num==40:
                               labels.append(p[0])
                               labels.append(p[1])
                               total+=1
                           #依序是右眼左邊右眼瞳孔右眼右邊
                           elif num==90 or num==89 or num==94:
                               labels.append(p[0])
                               labels.append(p[1])
                               total+=1
                           #依序是鼻子左邊鼻子瞳孔鼻子右邊
                           elif num==78 or num==87 or num==84:
                               labels.append(p[0])
                               labels.append(p[1])
                               total+=1           
                    labels.append(coord[num1])
                    labels.append(coord[num1+length])
                    num1+=1
                    writer.writerow(labels)   
    print(i) 

            
                        
          
               
    	  
    	  
    	  	
    	  	
    	  	
    
   
