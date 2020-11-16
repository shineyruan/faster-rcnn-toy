# sys modules
import numpy as np
import torch
from functools import partial
import cv2


def visual_bbox_mask(image: torch.Tensor, bboxes: torch.Tensor,
                     labels: torch.Tensor, scores: torch.Tensor = None,
                     gt_bbox: torch.Tensor = None, bbox_format='corner'):
    """
    Input:
        image:  tensor, (3, H, W)           image, on CPU
        bboxes: tensor, (num_obj, 4)        per image bounding boxes, on CPU
        scores: tensor, (num_obj, )         per image class confidence scores, on CPU
        bbox_format:    'corner' - [x1, y1, x2, y2]; 'center' - [x, y, w, h]
    """
    out_img = np.copy(image.numpy().transpose(1, 2, 0))
    out_img = (out_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    out_img = out_img.astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    for i in range(bboxes.shape[0]):
        if bbox_format == 'corner':
            x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
        else:
            x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

        color = [0, 0, 0]
        if labels is not None:
            color[labels[i]] = 255
        color = tuple(color)

        out_img = cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        if scores is not None:
            out_img = cv2.putText(out_img,
                                  "Score: {}".format(scores[i]),
                                  (int(x1), int(y1)),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  1.0,
                                  (20, 200, 200),
                                  2)

    if gt_bbox is not None:
        for i in range(gt_bbox.shape[0]):
            if bbox_format == 'corner':
                x1_gt, y1_gt, x2_gt, y2_gt = \
                    gt_bbox[i][0], gt_bbox[i][1], gt_bbox[i][2], gt_bbox[i][3]
            else:
                x_gt, y_gt, w_gt, h_gt = gt_bbox[i][0], gt_bbox[i][1], gt_bbox[i][2], gt_bbox[i][3]
                x1_gt = x_gt - w_gt / 2
                y1_gt = y_gt - h_gt / 2
                x2_gt = x_gt + w_gt / 2
                y2_gt = y_gt + h_gt / 2

            out_img = cv2.rectangle(out_img, (int(x1_gt), int(y1_gt)),
                                    (int(x2_gt), int(y2_gt)), (255, 255, 255), 3)

    return out_img


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)

    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes


def IOU(boxA, boxB, mode='center'):
    ##################################
    # compute the IOU between the boxA, boxB boxes
    # x_center, y_center, w, h
    ##################################
    if mode == 'center':
        inter_x1 = max(boxA[0] - boxA[2] / 2, boxB[0] - boxB[2] / 2)
        inter_y1 = max(boxA[1] - boxA[3] / 2, boxB[1] - boxB[3] / 2)
        inter_x2 = min(boxA[0] + boxA[2] / 2, boxB[0] + boxB[2] / 2)
        inter_y2 = min(boxA[1] + boxA[3] / 2, boxB[1] + boxB[3] / 2)
        area_boxA = boxA[2] * boxA[3]
        area_boxB = boxB[2] * boxB[3]
    else:
        inter_x1 = max(boxA[0], boxB[0])
        inter_y1 = max(boxA[1], boxB[1])
        inter_x2 = min(boxA[2], boxB[2])
        inter_y2 = min(boxA[3], boxB[3])
        area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        area_boxB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    inter = max((inter_x2 - inter_x1), 0) * max((inter_y2 - inter_y1), 0)
    iou = inter / (area_boxA + area_boxB - inter + 1)
    return iou


def matrix_IOU_center(boxA, boxB, device='cpu'):
    ##################################
    # compute the IOU between the boxA, boxB boxes
    # box: (grid_x, grid_y, 4)
    ##################################
    inter_x1 = torch.max(boxA[..., 0] - boxA[..., 2] / 2, boxB[..., 0] - boxB[..., 2] / 2)
    inter_y1 = torch.max(boxA[..., 1] - boxA[..., 3] / 2, boxB[..., 1] - boxB[..., 3] / 2)
    inter_x2 = torch.min(boxA[..., 0] + boxA[..., 2] / 2, boxB[..., 0] + boxB[..., 2] / 2)
    inter_y2 = torch.min(boxA[..., 1] + boxA[..., 3] / 2, boxB[..., 1] + boxB[..., 3] / 2)
    inter = torch.max((inter_x2 - inter_x1), torch.zeros(inter_x2.shape).to(device)) * \
        torch.max((inter_y2 - inter_y1), torch.zeros(inter_x2.shape).to(device))
    iou = inter / (boxA[..., 2] * boxA[..., 3] + boxB[..., 2] * boxB[..., 3] - inter + 1)
    return iou


def matrix_IOU_corner(boxA, boxB, device='cpu'):
    ##################################
    # compute the IOU between the boxA, boxB boxes
    # box: (grid_x, grid_y, 4)([x1, y1, x2, y2] format)
    ##################################
    inter_x1 = torch.max(boxA[..., 0], boxB[..., 0])
    inter_y1 = torch.max(boxA[..., 1], boxB[..., 1])
    inter_x2 = torch.min(boxA[..., 2], boxB[..., 2])
    inter_y2 = torch.min(boxA[..., 3], boxB[..., 3])
    inter = torch.max((inter_x2 - inter_x1), torch.zeros(inter_x2.shape).to(device)) * \
        torch.max((inter_y2 - inter_y1), torch.zeros(inter_x2.shape).to(device))
    iou = inter / ((boxA[..., 2] - boxA[..., 0]) * (boxA[..., 3] - boxA[..., 1]) +
                   (boxB[..., 2] - boxB[..., 0]) * (boxB[..., 3] - boxB[..., 1]) - inter + 1)
    return iou


def corner_to_center(bbox):
    # bbox: (4, ) [x1, y1, x2, y2] format
    # output: (4, ) [x, y, w, h] format
    return torch.tensor([(bbox[0] + bbox[2]) / 2,
                         (bbox[1] + bbox[3]) / 2,
                         (bbox[2] - bbox[0]),
                         (bbox[3] - bbox[1])])


def corners_to_centers(bbox):
    # bbox: (n, 4) [x1, y1, x2, y2] format
    # output: (n, 4) [x, y, w, h] format
    output = torch.zeros(bbox.shape)
    output[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0
    output[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0
    output[:, 2] = bbox[:, 2] - bbox[:, 0]
    output[:, 3] = bbox[:, 3] - bbox[:, 1]
    return output


def center_to_corner(bbox):
    # bbox: (4, ) [x, y, w, h] format
    # output: (4, ) [x1, y1, x2, y2] format
    return torch.tensor([bbox[0] - bbox[2] / 2,
                         bbox[1] - bbox[3] / 2,
                         bbox[0] + bbox[2] / 2,
                         bbox[1] + bbox[3] / 2])


def centers_to_corners(bbox):
    # bbox: (n, 4) [x, y, w, h] format
    # output: (n, 4) [x1, y1, x2, y2] format
    output = torch.zeros(bbox.shape)
    output[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    output[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    output[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
    output[:, 3] = bbox[:, 1] + bbox[:, 3] / 2
    return output


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)


def output_decodingd(regressed_boxes_t, flatten_proposals, device='cpu'):

    return box
