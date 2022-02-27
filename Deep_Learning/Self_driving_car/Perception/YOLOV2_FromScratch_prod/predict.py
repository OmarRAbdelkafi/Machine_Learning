import numpy as np
import cv2, copy
import seaborn as sns
import matplotlib.pyplot as plt    # for plotting the images
import os

import EDA as EDA

ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                    2.71054693,  5.12469308,  # anchor box 2, width,  height
                    10.47181473, 10.09646365,  # anchor box 3, width,  height
                    5.48531347,  8.11011331]) # anchor box 4, width,  height

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']


#batch generation of 16 images
BATCH_SIZE       = 16
GRID_H,  GRID_W  = 13 , 13
IMAGE_H, IMAGE_W = 416, 416
TRUE_BOX_BUFFER  = 50
BOX = int(len(ANCHORS)/2)
CLASS = len(LABELS)

generator_config = {
        'IMAGE_H'         : IMAGE_H,
        'IMAGE_W'         : IMAGE_W,
        'GRID_H'          : GRID_H,
        'GRID_W'          : GRID_W,
        'BOX'             : BOX,
        'LABELS'          : LABELS,
        'ANCHORS'         : ANCHORS,
        'BATCH_SIZE'      : BATCH_SIZE,
        'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

class OutputRescaler(object):
    def __init__(self,ANCHORS):
        self.ANCHORS = ANCHORS

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

    def get_shifting_matrix(self,netout):

        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[...,0]

        ANCHORSw = self.ANCHORS[::2]
        ANCHORSh = self.ANCHORS[1::2]

        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:,igrid_w,:] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h,:,:] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):
            mat_ANCHOR_W[:,:,ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no)
        for ianchor in range(BOX):
            mat_ANCHOR_H[:,:,ianchor] = ANCHORSh[ianchor]
        return(mat_GRID_W,mat_GRID_H,mat_ANCHOR_W,mat_ANCHOR_H)

    def fit(self, netout):
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)

        a single image output of model.predict()
        '''
        GRID_H, GRID_W, BOX = netout.shape[:3]

        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = self.get_shifting_matrix(netout)


        # bounding box parameters
        netout[..., 0]   = (self._sigmoid(netout[..., 0]) + mat_GRID_W)/GRID_W # x      unit: range between 0 and 1
        netout[..., 1]   = (self._sigmoid(netout[..., 1]) + mat_GRID_H)/GRID_H # y      unit: range between 0 and 1
        netout[..., 2]   = (np.exp(netout[..., 2]) * mat_ANCHOR_W)/GRID_W      # width  unit: range between 0 and 1
        netout[..., 3]   = (np.exp(netout[..., 3]) * mat_ANCHOR_H)/GRID_H      # height unit: range between 0 and 1
        # rescale the confidence to range 0 and 1
        netout[..., 4]   = self._sigmoid(netout[..., 4])
        expand_conf      = np.expand_dims(netout[...,4],-1) # (N grid h , N grid w, N anchor , 1)
        # rescale the class probability to range between 0 and 1
        # Pr(object class = k) = Pr(object exists) * Pr(object class = k |object exists)
        #                      = Conf * P^c
        netout[..., 5:]  = expand_conf * self._softmax(netout[..., 5:])
        # ignore the class probability if it is less than obj_threshold

        return(netout)

def find_high_class_probability_bbox(netout_scale, obj_threshold):
    '''
    == Input ==
    netout : y_pred[i] np.array of shape (GRID_H, GRID_W, BOX, 4 + 1 + N class)

             x, w must be a unit of image width
             y, h must be a unit of image height
             c must be in between 0 and 1
             p^c must be in between 0 and 1
    == Output ==

    boxes  : list containing bounding box with Pr(object is in class C) > 0 for at least in one class C


    '''
    GRID_H, GRID_W, BOX = netout_scale.shape[:3]

    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scale[row,col,b,5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scale[row,col,b,:4]
                    confidence = netout_scale[row,col,b,4]
                    box = EDA.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return(boxes)

def draw_boxes(image, boxes, labels, verbose):
    '''
    image : np.array of shape (N height, N width, 3)
    '''
    def adjust_minmax(c,_max):
        if c < 0:
            c = 0
        if c > _max:
            c = _max
        return c

    obj_baseline=0.005
    image = copy.deepcopy(image)
    image_h, image_w, _ = image.shape
    score_rescaled  = np.array([box.get_score() for box in boxes])
    score_rescaled /= obj_baseline

    colors = sns.color_palette("husl", 8)

    for sr, box, color in zip(score_rescaled, boxes, colors):
        xmin = adjust_minmax(int(box.xmin*image_w),image_w)
        ymin = adjust_minmax(int(box.ymin*image_h),image_h)
        xmax = adjust_minmax(int(box.xmax*image_w),image_w)
        ymax = adjust_minmax(int(box.ymax*image_h),image_h)

        text = "{:10} {:4.3f}".format(labels[box.label], box.get_score())

        if verbose:
            print("{} xmin={:4.0f},ymin={:4.0f},xmax={:4.0f},ymax={:4.0f}".format(text,xmin,ymin,xmax,ymax))

        sr = int(sr // 1)
        if sr <= 0 :
            sr = 1

        cv2.rectangle(image, pt1=(xmin,ymin), pt2=(xmax,ymax), color=color, thickness=sr)

        cv2.putText(img       = image,
                    text      = text,
                    org       = (xmin+ 13, ymin + 13),
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1e-3 * image_h,
                    color     = (1, 0, 1),
                    thickness = 1)

    return image

def nonmax_suppression(boxes, iou_threshold, obj_threshold):
    '''
    boxes : list containing "good" BoundBox of a frame
            [BoundBox(),BoundBox(),...]
    '''
    bestAnchorBoxFinder    = EDA.BestAnchorBoxFinder([])

    CLASS    = len(boxes[0].classes)
    index_boxes = []
    # suppress non-maximal boxes
    for c in range(CLASS):
        # extract class probabilities of the c^th class from multiple bbox
        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        #sorted_indices[i] contains the i^th largest class probabilities
        sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            # if class probability is zero then ignore
            if boxes[index_i].classes[c] == 0:
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                    # if so, the remaining bbox' class probabilities are set to 0.
                    bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)

    newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]

    return newboxes

def predict_batch_image(train_image_folder, model):
    np.random.seed(1)
    Nsample   = 2
    image_nms = list(np.random.choice(os.listdir(train_image_folder),Nsample))
    iou_threshold = 0.5
    obj_threshold = 0.04

    outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
    imageReader = EDA.ImageReader(IMAGE_H, IMAGE_W, norm=EDA.normalize)
    X_test = []

    for img_nm in image_nms:
        _path    = os.path.join(train_image_folder,img_nm)
        out      = imageReader.fit(_path)
        X_test.append(out)

    X_test = np.array(X_test)

    ## model
    dummy_array    = np.zeros((len(X_test),1,1,1,TRUE_BOX_BUFFER,4))
    y_pred         = model.predict([X_test,dummy_array])

    for iframe in range(len(y_pred)):
        netout         = y_pred[iframe]
        netout_scale   = outputRescaler.fit(netout)
        boxes          = find_high_class_probability_bbox(netout_scale, obj_threshold)
        if len(boxes) > 0:
            final_boxes    = nonmax_suppression(boxes, iou_threshold, obj_threshold)
            ima = draw_boxes(X_test[iframe],final_boxes,LABELS,verbose=True)
            figsize = (20,20)
            plt.figure(figsize=figsize)
            plt.imshow(ima);
            plt.show()
