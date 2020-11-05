# sys modules
import numpy as np
import torch
from functools import partial
import cv2


def visual_bbox_mask(image, masks=None, bboxs=None, labels=None, bbox_format='corner'):
    """
    Input:
        image: tensor, 3 * h * w
        masks: tensor, num_obj * h * w
        bbox_format:    'corner' - [x1, y1, x2, y2]; 'center' - [x,y,w,h]
    """
    outim = np.copy(image.cpu().numpy().transpose(1, 2, 0))
    outim = (outim * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    outim = outim.astype(np.uint8)
    outim = cv2.cvtColor(outim, cv2.COLOR_RGB2BGR)
    if bboxs is not None:
        for i in range(bboxs.shape[0]):
            if bbox_format == 'corner':
                x1, y1, x2, y2 = bboxs[i][0], bboxs[i][1], bboxs[i][2], bboxs[i][3]
            else:
                x, y, w, h = bboxs[i][0], bboxs[i][1], bboxs[i][2], bboxs[i][3]
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2

            if labels is not None:
                if labels[i] == 0:
                    color = (255, 0, 0)
                elif labels[i] == 1:
                    color = (0, 255, 0)
                elif labels[i] == 2:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)
            else:
                color = (255, 0, 0)
            outim = cv2.rectangle(outim, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    if masks is not None:
        for i in range(masks.shape[0]):
            outim = outim.astype(np.uint32)
            if labels is None:
                outim[:, :, (i + 2) % 3] = \
                    np.clip(outim[:, :, (i + 2) % 3] + masks[i].cpu().numpy() * 100, 0, 255)
            else:
                outim[:, :, labels[i]] = \
                    np.clip(outim[:, :, labels[i]] + masks[i].cpu().numpy() * 100, 0, 255)
            outim = outim.astype(np.uint8)
            if labels is not None and bboxs is not None:
                outim = cv2.putText(outim,
                                    "Class: {}".format(labels[i]),
                                    (int(bboxs[i][0]), int(bboxs[i][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (20, 200, 200),
                                    2)
    return outim


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)

    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes


def IOU(boxA, boxB):

    return iou


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t, flatten_proposals, device='cpu'):

    return box
