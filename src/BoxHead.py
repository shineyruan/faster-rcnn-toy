import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from sklearn.metrics import auc

from utils import (
    matrix_IOU_corner, corner_to_center, center_to_corner, corners_to_centers, centers_to_corners,
    IOU
)


class BoxHead(torch.nn.Module):
    def __init__(self, device='cuda', Classes=3, P=7):
        super(BoxHead, self).__init__()

        self.device = device
        self.C = Classes
        self.P = P

        # initialize BoxHead
        self.intermediate = nn.Sequential(nn.Linear(in_features=256 * P * P, out_features=1024),
                                          nn.ReLU(),
                                          nn.Linear(in_features=1024, out_features=1024),
                                          nn.ReLU())

        self.class_head = nn.Linear(in_features=1024, out_features=self.C + 1)
        self.softmax = nn.Softmax(dim=1)  # only used in testing mode

        self.reg_head = nn.Linear(in_features=1024, out_features=4 * self.C)

    def create_ground_truth(self, proposals: list, gt_labels: list, gt_bboxes: list):
        """
        This function assigns to each proposal either a ground truth box or the background class
        (we assume background class is 0)
        Input:
        -----
            proposals: list: len(bz){(per_image_proposals, 4)}([x1, y1, x2, y2] format)
            gt_labels: list: len(bz) {(n_obj)}
            gt_bboxes: list: len(bz){(n_obj, 4)}([x1, y1, x2, y2] format)
        Output:
        -----
            (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals, 1)(the class that the proposal is assigned)
            regressor_target: (total_proposals, 4)(target encoded in the[t_x, t_y, t_w, t_h] format)
        """
        labels = []
        regressor_target = []
        for batch_id in range(len(gt_labels)):
            num_proposals = proposals[batch_id].shape[0]
            num_gtbboxes = gt_bboxes[batch_id].shape[0]
            proposal_mat = proposals[batch_id].expand(
                num_gtbboxes, num_proposals, 4).to(self.device)
            gtbbox_mat = gt_bboxes[batch_id].expand(num_proposals, num_gtbboxes, 4)
            gtbbox_mat = torch.transpose(gtbbox_mat, 0, 1).to(self.device)
            # iou_mat: (num_gtbboxes, num_proposals)
            iou_mat = matrix_IOU_corner(proposal_mat, gtbbox_mat, device=self.device)

            matched_gtbbox = torch.max(iou_mat, dim=0)
            background = matched_gtbbox.values < 0.5

            proposal_labels = torch.zeros(num_proposals, 1).to(self.device)
            regressor = torch.zeros(num_proposals, 4).to(self.device)
            for i in range(num_proposals):
                if not background[i]:
                    proposal_labels[i, 0] = gt_labels[batch_id][matched_gtbbox.indices[i]]

                gt_bbox_center = corner_to_center(gt_bboxes[batch_id][matched_gtbbox.indices[i]])
                proposal_center = corner_to_center(proposals[batch_id][i])
                regressor[i, 0] = (gt_bbox_center[0] - proposal_center[0]) / proposal_center[2]
                regressor[i, 1] = (gt_bbox_center[1] - proposal_center[1]) / proposal_center[3]
                regressor[i, 2] = torch.log(gt_bbox_center[2] / proposal_center[2])
                regressor[i, 3] = torch.log(gt_bbox_center[3] / proposal_center[3])

            labels.append(proposal_labels)
            regressor_target.append(regressor)

        labels = torch.cat(labels, dim=0)
        regressor_target = torch.cat(regressor_target, dim=0)

        return labels, regressor_target

    def MultiScaleRoiAlign(self, fpn_feat_list: list, proposals: list,
                           P: int = 7, img_size: tuple = (800, 1088)) -> torch.Tensor:
        """
        This function for each proposal finds the appropriate feature map to sample and using
        RoIAlign it samples a (256,P,P) feature map. This feature map is then flattened into a
        (256*P*P) vector
        Input:
        ------
             fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
             proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
             P: scalar
        Output:
        ------
             feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals
                                                            are the same as the ground truth
                                                            creation)
        """

        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################

        fpn_names = [str(i) for i in range(len(fpn_feat_list))]
        fpn_dict = {}
        for i, name in enumerate(fpn_names):
            fpn_dict[name] = fpn_feat_list.pop(0)

        del fpn_feat_list

        roi_fn = torchvision.ops.MultiScaleRoIAlign(fpn_names, (P, P), sampling_ratio=-1)
        feature_vectors = roi_fn(fpn_dict, proposals, [img_size])

        return torch.reshape(feature_vectors, (feature_vectors.shape[0], 256 * P * P))

    def postprocess_detections(self, class_logits, box_regression, proposals,
                               conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50,
                               img_size: tuple = (800, 1088)):
        """
        This function does the post processing for the results of the Box Head for a batch of images
        Use the proposals to distinguish the outputs from each image
        Input:
        -----
              class_logits: (total_proposals,(C+1))
              box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
              proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN
                                                                [x1,y1,x2,y2] format)
              conf_thresh: scalar
              keep_num_preNMS: scalar (number of boxes to keep pre NMS)
              keep_num_postNMS: scalar (number of boxes to keep post NMS)
        Output:
        -----
              boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
              scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for
                                                                    the regressed box)
              labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        """
        boxes = []
        scores = []
        labels = []

        start_id = 0
        for batch_id in range(len(proposals)):
            per_image_proposals = proposals[batch_id]
            per_image_class = class_logits[start_id:start_id + per_image_proposals.shape[0], :]
            per_image_box = box_regression[start_id:start_id + per_image_proposals.shape[0], :]
            start_id += per_image_proposals.shape[0]

            # remove background boxes
            conf_thresh_tensor = conf_thresh * \
                torch.ones(per_image_class[:, 1].shape).to(self.device)
            to_remove_id = \
                (per_image_class[:, 1] < conf_thresh_tensor) \
                * (per_image_class[:, 2] < conf_thresh_tensor) \
                * (per_image_class[:, 3] < conf_thresh_tensor)
            # print(to_remove_id)
            if torch.sum(~to_remove_id) == 0:
                boxes.append(torch.zeros(1, 4))
                scores.append(torch.zeros(1))
                labels.append(torch.zeros(1, dtype=torch.long))
                continue
            per_image_proposals = per_image_proposals[~to_remove_id]
            per_image_class = per_image_class[~to_remove_id]
            per_image_box = per_image_box[~to_remove_id]
            per_image_label = torch.argmax(per_image_class, dim=1) - 1
            # print(per_image_label)

            # transform to [x1,y1,x2,y2] format
            per_image_proposals_center = corners_to_centers(per_image_proposals).to(self.device)
            prediction_box_center = torch.zeros(per_image_box.shape).to(self.device)
            for i in range(3):
                prediction_box_center[:, 0 + i * 4] = per_image_box[:, 0 + i * 4] * \
                    per_image_proposals_center[:, 2] + per_image_proposals_center[:, 0]
                prediction_box_center[:, 1 + i * 4] = per_image_box[:, 1 + i * 4] * \
                    per_image_proposals_center[:, 3] + per_image_proposals_center[:, 1]
                prediction_box_center[:, 2 + i * 4] \
                    = torch.exp(per_image_box[:, 2 + i * 4]) * per_image_proposals_center[:, 2]
                prediction_box_center[:, 3 + i * 4] \
                    = torch.exp(per_image_box[:, 3 + i * 4]) * per_image_proposals_center[:, 3]
            prediction_box_corner = torch.zeros(per_image_box.shape).to(self.device)
            for i in range(3):
                prediction_box_corner[:, i * 4:i * 4 + 4] \
                    = centers_to_corners(prediction_box_center[:, i * 4:i * 4 + 4]).to(self.device)
                # Crop the x1, x2
                prediction_box_corner[:, i * 4] = \
                    torch.min(prediction_box_corner[:, i * 4],
                              img_size[1] * torch.ones(prediction_box_corner[:, i * 4].shape)
                              .to(self.device))
                prediction_box_corner[:, i * 4 + 2] = \
                    torch.min(prediction_box_corner[:, i * 4 + 2],
                              img_size[1] * torch.ones(prediction_box_corner[:, i * 4].shape)
                              .to(self.device))
                # Crop the y1, y2
                prediction_box_corner[:, i * 4 + 1] = \
                    torch.min(prediction_box_corner[:, i * 4 + 1],
                              img_size[0] * torch.ones(prediction_box_corner[:, i * 4].shape)
                              .to(self.device))
                prediction_box_corner[:, i * 4 + 3] = \
                    torch.min(prediction_box_corner[:, i * 4 + 3],
                              img_size[0] * torch.ones(prediction_box_corner[:, i * 4].shape)
                              .to(self.device))
            prediction_box_corner = torch.max(prediction_box_corner,
                                              torch.zeros(prediction_box_corner.shape)
                                              .to(self.device))

            prediction_box_corner_max = torch.zeros(
                prediction_box_corner.shape[0], 4).to(self.device)
            prediction_score_max = torch.max(per_image_class, dim=1).values.to(self.device)
            # print(prediction_score_max)
            for i in range(prediction_box_corner_max.shape[0]):
                prediction_box_corner_max[i, :] = \
                    prediction_box_corner[i, per_image_label[i] * 4:per_image_label[i] * 4 + 4]
            # print(prediction_box_corner_max)

            # Keep the proposals with the top K objectness scores
            prediction_score_max_sorted_idx = torch.argsort(prediction_score_max, descending=True)
            prediction_score_max_K = \
                prediction_score_max[prediction_score_max_sorted_idx[:keep_num_preNMS]]
            prediction_box_corner_K = \
                prediction_box_corner_max[prediction_score_max_sorted_idx[:keep_num_preNMS]]
            prediction_label_K = per_image_label[prediction_score_max_sorted_idx[:keep_num_preNMS]]
            # print(prediction_score_max_K)

            # Compute NMS
            nms_predict_scores = self.NMS(prediction_score_max_K, prediction_box_corner_K)

            #   pick the top N mat classes & proposal coords after NMS
            nms_sorted_idx = torch.argsort(nms_predict_scores, descending=True)
            nms_labels = prediction_label_K[nms_sorted_idx[:keep_num_postNMS]]
            nms_prebox = prediction_box_corner_K[nms_sorted_idx[:keep_num_postNMS]]
            nms_scores = prediction_score_max_K[nms_sorted_idx[:keep_num_postNMS]]

            boxes.append(nms_prebox)
            scores.append(nms_scores)
            labels.append(nms_labels)

        return boxes, scores, labels

    def get_top_K(self, class_logits, box_regression, proposals, k=20):
        """
        Input:
        -----
              class_logits: (total_proposals,(C+1))
              box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
              proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN
                                                                [x1,y1,x2,y2] format)
        Output:
        -----
              boxes: list:len(bz){(k, 4)}   ([x1,y1,x2,y2] format)
              scores: list:len(bz){(k, )}   (the score for the top class for the regressed box)
              labels: list:len(bz){(k, )}   (top class of each regressed box)
        """
        boxes = []
        scores = []
        labels = []

        start_id = 0
        for batch_id in range(len(proposals)):
            per_image_proposals = proposals[batch_id]
            per_image_class = class_logits[start_id:start_id + per_image_proposals.shape[0], :]
            per_image_box = box_regression[start_id:start_id + per_image_proposals.shape[0], :]
            start_id += per_image_proposals.shape[0]

            per_image_label = torch.argmax(per_image_class, dim=1) - 1
            bg_ids = per_image_label == -1

            if torch.sum(~bg_ids) == 0:
                boxes.append(torch.zeros(1, 4))
                scores.append(torch.zeros(1))
                labels.append(torch.zeros(1, dtype=torch.long))
                continue

            per_image_proposals_fg = per_image_proposals[~bg_ids]
            per_image_class_fg = per_image_class[~bg_ids]
            per_image_box_fg = per_image_box[~bg_ids]
            per_image_label_fg = per_image_label[~bg_ids]

            per_image_proposals_fg_center = \
                corners_to_centers(per_image_proposals_fg).to(self.device)
            box_fg = torch.zeros(per_image_box_fg.shape[0], 4).to(self.device)

            for i in range(box_fg.shape[0]):
                label = per_image_label_fg[i]
                box_fg[:, 0] = per_image_box_fg[:, 0 + label * 4] * \
                    per_image_proposals_fg_center[:, 2] + per_image_proposals_fg_center[:, 0]
                box_fg[:, 1] = per_image_box_fg[:, 1 + label * 4] * \
                    per_image_proposals_fg_center[:, 3] + per_image_proposals_fg_center[:, 1]
                box_fg[:, 2] \
                    = torch.exp(per_image_box_fg[:, 2 + label * 4]) * per_image_proposals_fg_center[:, 2]
                box_fg[:, 3] \
                    = torch.exp(per_image_box_fg[:, 3 + label * 4]) * per_image_proposals_fg_center[:, 3]
            box_fg_corner = centers_to_corners(box_fg)

            score_fg = torch.max(per_image_class_fg, dim=1).values.to(self.device)
            score_fg_sorted_idx = torch.argsort(score_fg, descending=True)
            boxes.append(box_fg_corner[score_fg_sorted_idx[:k]])
            scores.append(score_fg[score_fg_sorted_idx[:k]])
            labels.append(per_image_label_fg[score_fg_sorted_idx[:k]])

        return boxes, scores, labels

    def NMS(self, predict_class, prebox):
        """
        Input:
        -----
              predict_class: (top_k_boxes) (scores of the top k boxes)
              prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        Output:
        -----
              nms_predict_class: (Post_NMS_boxes)
        """

        ##################################
        # perform NMS
        ##################################

        K = prebox.shape[0]

        # compute the IOU for bounding boxes
        bbox_mesh_x = prebox.repeat(K, 1, 1)
        bbox_mesh_y = bbox_mesh_x.permute(1, 0, 2)

        ious = matrix_IOU_corner(bbox_mesh_x, bbox_mesh_y, device=self.device).triu(diagonal=1)

        # IOU(*, i) for (n_act, n_act)
        #   take i as row and k as column
        #   for each i, all non-zero (i, k) represents IOU(i, k) where s_k > s_i
        #
        # ious_i == IOU(*, i)
        #
        #   since we would like to minimize f(IOU(*, i)), we wish to find max(IOU(*, i))
        #
        ious_i = torch.max(ious, dim=0).values
        ious_i = ious_i.expand(K, K).T

        # Matrix NMS
        decay = (1 - ious) / (1 - ious_i)

        # min of f(IOU(i, j)) / f(IOU(*, i))
        decay = torch.min(decay, dim=0).values
        return decay * predict_class

    def classifier_loss(self, class_minibatch: torch.Tensor,
                        class_minibatch_gt: torch.Tensor) -> float:
        """
        Compute the loss of the classifier
        Input:
        -----
            class_minibatch: (mini_batch,(C+1))
            class_minibatch_gt: (mini_batch,1)
        Output:
        ----
            loss_class:  scalar
        """
        ceLoss_fn = nn.CrossEntropyLoss()
        return ceLoss_fn(class_minibatch, class_minibatch_gt)

    def regression_loss(self, box_preds: torch.Tensor, regression_targets: torch.Tensor,
                        class_minibatch_gt: torch.Tensor) -> float:
        """
        Compute the regression loss of the regressor
        Input:
        -----
            box_preds:              (mini_batch,4*C)
            regression_targets:     (mini_batch,4)
            class_minibatch_gt:     only for reference to find the corresponding bbox preds
        Output:
        ----
            loss_regr:  scalar
        """

        loss_regr = torch.tensor(0.0).to(self.device)
        count = 0
        sl1loss = nn.SmoothL1Loss(reduction='sum')

        for i in range(class_minibatch_gt.shape[0]):
            if class_minibatch_gt[i] > 0:
                label = class_minibatch_gt[i] - 1
                pred = box_preds[i][4 * label:4 * label + 4]
                target = regression_targets[i]
                loss_regr += sl1loss(pred, target)
            count += 1

        return loss_regr / count

    def compute_loss(self, class_logits: torch.Tensor, box_preds: torch.Tensor,
                     labels: torch.Tensor, regression_targets: torch.Tensor,
                     lambda_coeff=1, effective_batch=150,
                     rand_perm_fg=None,
                     rand_perm_bg=None) -> tuple:
        """
        Compute the total loss of the classifier and the regressor
        Input:
        -----
             class_logits: (total_proposals,(C+1)) (as outputted from forward, not passed from
                                                    softmax so we can use CrossEntropyLoss)
             box_preds: (total_proposals,4*C)      (as outputted from forward)
             labels: (total_proposals,1)
             regression_targets: (total_proposals,4)
             lambda_coeff: scalar (weighting of the two losses)
             effective_batch: scalar
        Outpus:
        -----
             loss: scalar
             loss_class: scalar
             loss_regr: scalar
        """

        # For sampling the mini batch we can use a similar policy with the RPN but now we sample
        # proposals with no-background ground truth and proposals with background ground truth with
        # a ratio as close to 3:1 as possible. Again you should set a constant size for the
        # mini-batch.

        assert class_logits.shape[-1] == self.C + 1

        class_minibatch = torch.Tensor([]).to(self.device)
        class_minibatch_gt = torch.tensor([]).to(self.device)
        reg_minibatch = torch.Tensor([]).to(self.device)
        reg_minibatch_gt = torch.Tensor([]).to(self.device)

        no_bg_idx = (labels.T.squeeze() > 0).nonzero(as_tuple=False).squeeze()
        bg_idx = (labels.T.squeeze() == 0).nonzero(as_tuple=False).squeeze()

        if rand_perm_fg is None and rand_perm_bg is None:
            M_bg = int(effective_batch / 4)
            M_no_bg = effective_batch - M_bg

            if torch.sum(labels != 0) > M_no_bg:
                # Randomly choose M/2 anchors with positive ground truth labels.
                rand_idx = torch.randint(no_bg_idx.size()[0], (M_no_bg,))
                class_minibatch = class_logits[no_bg_idx[rand_idx]].reshape(-1, self.C + 1)
                class_minibatch_gt = labels[no_bg_idx[rand_idx]].reshape(-1, 1)
                reg_minibatch = box_preds[no_bg_idx[rand_idx]].reshape(-1, 4 * self.C)
                reg_minibatch_gt = regression_targets[no_bg_idx[rand_idx]].reshape(-1, 4)
            else:
                class_minibatch = class_logits[no_bg_idx].reshape(-1, self.C + 1)
                class_minibatch_gt = labels[no_bg_idx].reshape(-1, 1)
                reg_minibatch = box_preds[no_bg_idx].reshape(-1, 4 * self.C)
                reg_minibatch_gt = regression_targets[no_bg_idx].reshape(-1, 4)

            if torch.sum(labels == 0) > M_bg:
                rand_idx = torch.randint(bg_idx.size()[0], (M_bg,))
                class_minibatch = torch.cat(
                    [class_minibatch,
                     class_logits[bg_idx[rand_idx]].reshape(-1, self.C + 1)],
                    dim=0)
                class_minibatch_gt = torch.cat(
                    [class_minibatch_gt,
                     labels[bg_idx[rand_idx]].reshape(-1, 1)],
                    dim=0)
                reg_minibatch = torch.cat(
                    [reg_minibatch,
                     box_preds[bg_idx[rand_idx]].reshape(-1, 4 * self.C)],
                    dim=0)
                reg_minibatch_gt = torch.cat(
                    [reg_minibatch_gt,
                     regression_targets[bg_idx[rand_idx]].reshape(-1, 4)],
                    dim=0)
            else:
                class_minibatch = torch.cat(
                    [class_minibatch,
                     class_logits[bg_idx].reshape(-1, self.C + 1)],
                    dim=0)
                class_minibatch_gt = torch.cat(
                    [class_minibatch_gt,
                     labels[bg_idx].reshape(-1, 1)],
                    dim=0)
                reg_minibatch = torch.cat(
                    [reg_minibatch,
                     box_preds[bg_idx].reshape(-1, 4 * self.C)],
                    dim=0)
                reg_minibatch_gt = torch.cat(
                    [reg_minibatch_gt,
                     regression_targets[bg_idx].reshape(-1, 4)],
                    dim=0)

        else:
            class_minibatch = class_logits[no_bg_idx][rand_perm_fg].reshape(-1, self.C + 1)
            class_minibatch_gt = labels[no_bg_idx][rand_perm_fg].reshape(-1, 1)
            reg_minibatch = box_preds[no_bg_idx][rand_perm_fg].reshape(-1, 4 * self.C)
            reg_minibatch_gt = regression_targets[no_bg_idx][rand_perm_fg].reshape(-1, 4)

            class_minibatch = torch.cat(
                [class_minibatch,
                 class_logits[bg_idx][rand_perm_bg].reshape(-1, self.C + 1)],
                dim=0)
            class_minibatch_gt = torch.cat(
                [class_minibatch_gt,
                 labels[bg_idx][rand_perm_bg].reshape(-1, 1)],
                dim=0)
            reg_minibatch = torch.cat(
                [reg_minibatch,
                 box_preds[bg_idx][rand_perm_bg].reshape(-1, 4 * self.C)],
                dim=0)
            reg_minibatch_gt = torch.cat(
                [reg_minibatch_gt,
                 regression_targets[bg_idx][rand_perm_bg].reshape(-1, 4)],
                dim=0)

        class_minibatch_gt = class_minibatch_gt.T.squeeze().type(torch.long)

        loss_class = self.classifier_loss(class_minibatch, class_minibatch_gt)
        loss_regr = self.regression_loss(reg_minibatch, reg_minibatch_gt, class_minibatch_gt)

        loss = loss_class + lambda_coeff * loss_regr

        return loss, loss_class, loss_regr

    def forward(self, feature_vectors: torch.Tensor, training: bool = True) -> tuple:
        """
        Forward the pooled feature vectors through the intermediate layer and the classifier,
        regressor of the box head
        Input:
        -----
               feature_vectors: (total_proposals, 256*P*P)
               training:        bool            whether the network is in training mode
        Outputs:
        -----
               class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus
                                                        background, notice if you want to use
                                                      CrossEntropyLoss you should not pass the
                                                        output through softmax here)
               box_pred:     (total_proposals,4*C)
        """

        intermediate_out = self.intermediate(feature_vectors)
        class_logits = self.class_head(intermediate_out)
        if not training:
            class_logits = self.softmax(class_logits)
        box_pred = self.reg_head(intermediate_out)

        return class_logits, box_pred

    def box_head_evaluation(self, nms_boxes: list, nms_scores: list, nms_labels: list,
                            gt_boxes: list, gt_labels: list,
                            iou_thresh: float = 0.5) -> tuple:
        """
        Constructs matches list & scores list for every class

        Input
        -----
            nms_boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
            nms_scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for
                                                                the regressed box)
            nms_labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
            gt_bboxes: list: len(bz){(n_obj, 4)}([x1, y1, x2, y2] format)
            gt_labels: list: len(bz) {(n_obj)}

        Output
        -----
            matches:        (bz, post_NMS_boxes_per_image, 3)
            scores:         (bz, post_NMS_boxes_per_image, 3)
            num_true:       (3, )
            num_positives:  (3, )
        """

        matches = []
        scores = []
        num_trues = torch.zeros(1, 3)
        num_positives = torch.zeros(1, 3)

        batch_size = len(nms_labels)

        for bz in range(batch_size):
            match = torch.zeros(nms_labels[bz].shape[0], 3)
            score = torch.zeros(match.shape)
            num_true = torch.zeros(1, 3)
            num_positive = torch.zeros(num_true.shape)

            # calculate trues
            for obj_label in gt_labels[bz]:
                if obj_label > 0:
                    num_true[0, obj_label.type(torch.long) - 1] += 1

            # calculate positives
            for obj_label in nms_labels[bz]:
                num_positive[0, obj_label.type(torch.long)] += 1

            for i_pred, class_pred in enumerate(nms_labels[bz]):
                class_pred_scalar = class_pred.item()
                if class_pred_scalar + 1 in gt_labels[bz]:

                    # retrieve class gt label
                    i_gt_list = (gt_labels[bz] == class_pred + 1).nonzero(as_tuple=False).squeeze(0)

                    for i_gt in i_gt_list:
                        # retrieve masks
                        box_pred = nms_boxes[bz][i_pred]
                        box_gt = gt_boxes[bz][i_gt.item()]

                        # compute IOU
                        iou = IOU(box_pred, box_gt, mode='corner')

                        if iou > iou_thresh:
                            match[i_pred, class_pred] = 1

                # no matter how we always store the bbox scores
                score[i_pred, class_pred] = nms_scores[bz][i_pred]

            matches.append(match)
            scores.append(score)
            num_trues = torch.cat([num_trues, num_true], dim=0)
            num_positives = torch.cat([num_positives, num_positive], dim=0)

        return torch.cat(matches), torch.cat(scores), \
            torch.sum(num_trues, dim=0), torch.sum(num_positives, dim=0)

    def average_precision(self, match_values, score_values, total_trues, total_positives,
                          threshold=0.5):
        """
        Input:
        -----
            match_values - shape (N,) - matches with respect to true labels for a single class
            score_values - shape (N,) - objectness for a single class
            total_trues     - int      - total number of true labels for a single class in the
                                         entire dataset
            total_positives - int      - total number of positive labels for a single class in the
                                        entire dataset

        Output:
        -----
            area, sorted_recall, sorted_precision
        """
        # please fill in the appropriate arguments
        # compute the average precision as mentioned in the PDF.
        # it might be helpful to use - from sklearn.metrics import auc
        #   to compute area under the curve.
        area, sorted_recall, sorted_precision = None, None, None

        max_score = torch.max(score_values).item()
        ln = torch.linspace(threshold, max_score, steps=100)
        precision_mat = torch.zeros(101)
        recall_mat = torch.zeros(101)

        # iterate through the linspace
        for i, th in enumerate(ln):
            matches = match_values[score_values > th]
            TP = torch.sum(matches)  # true positives
            precision = 1
            if total_positives > 0:
                precision = TP / total_positives

            recall = 1
            if total_trues > 0:
                recall = TP / total_trues

            precision_mat[i] = precision
            recall_mat[i] = recall

        recall_mat[100] = 0
        precision_mat[100] = 1
        sorted_idx = torch.argsort(recall_mat)
        sorted_recall = recall_mat[sorted_idx]
        sorted_precision = precision_mat[sorted_idx]
        area = auc(sorted_recall, sorted_precision)

        return area, sorted_recall, sorted_precision


import os
if __name__ == '__main__':
    net = BoxHead(device='cuda', Classes=3, P=7)

    # Test Ground Truth creation
    # testcase 7 has one incorrect label because the iou of that one is 0.4999
    # testcase 3 has many incorrect regressor targets when running on cpu
    #       but no error when running on cuda
    for i in range(7):
        print("-------------------------", str(i), "-------------------------")
        testcase = torch.load("test/GroundTruth/ground_truth_test" + str(i) + ".pt")
        print(testcase.keys())
        print(len(testcase['bbox']))
        labels, regressor_target = net.create_ground_truth(testcase['proposals'],
                                                           testcase['gt_labels'],
                                                           testcase['bbox'])
        correctness = labels.cpu().type(
            torch.int8).reshape(-1) == testcase['labels'].type(torch.int8).reshape(-1)
        print(labels.type(torch.int8).reshape(-1)[~correctness])
        print(testcase['labels'].type(torch.int8).reshape(-1)[~correctness])
        correctness = torch.abs(testcase['regressor_target'] - regressor_target.cpu()) < 0.01
        # print((~correctness).nonzero())
        print(torch.abs(testcase['regressor_target'] - regressor_target.cpu())[~correctness])
        # print(testcase['regressor_target'][~correctness])
        # print(regressor_target[~correctness])

    # Test ROI align
    roi_dir = "test/MultiScaleRoiAlign/"
    for num_test in range(4):
        # load test cases
        path = os.path.join(roi_dir, "multiscale_RoIAlign_test" + str(num_test) + ".pt")
        fpn_feat_list = [item.cuda() for item in torch.load(path)['fpn_feat_list']]
        proposals = [item.cuda() for item in torch.load(path)['proposals']]
        output_feature_vectors = torch.load(path)['output_feature_vectors'].cuda()
        feature_vectors = net.MultiScaleRoiAlign(fpn_feat_list, proposals)
        print("\n----- ROI align test {} -----".format(num_test))
        print(feature_vectors)
        print(output_feature_vectors)
        correctness = torch.abs(feature_vectors - output_feature_vectors) < 0.01
        difference = torch.abs(feature_vectors - output_feature_vectors)[~correctness]
        print(difference, difference.shape)

    # Test Loss
    loss_dir = "test/Loss/"
    for i in range(7):
        print("\n----- Loss Align Test {} -----".format(i))
        path = os.path.join(loss_dir, "loss_test" + str(i) + ".pt")
        class_logits = torch.load(path)['clas_logits']
        bbox_preds = torch.load(path)['box_preds']
        labels = torch.load(path)['labels']
        regression_targets = torch.load(path)['regression_targets']
        effective_batch = torch.load(path)['effective_batch']
        random_permutation_foreground = torch.load(path)['random_permutation_foreground']
        random_permutation_background = torch.load(path)['random_permutation_background']
        loss_class_gt = torch.load(path)['loss_clas']
        loss_reg_gt = torch.load(path)['loss_reg']

        _, loss_class, loss_reg = \
            net.compute_loss(class_logits, bbox_preds,
                             labels, regression_targets,
                             effective_batch=effective_batch,
                             rand_perm_fg=random_permutation_foreground,
                             rand_perm_bg=random_permutation_background)

        print("loss_class_gt", loss_class_gt)
        print("loss_class   ", loss_class)
        print("loss_reg_gt  ", loss_reg_gt)
        print("loss_reg     ", loss_reg)

        correctness = torch.abs(loss_class - loss_class_gt) < 0.01
        difference = torch.abs(loss_class - loss_class_gt)[~correctness]
        print(difference, difference.shape)

    # Test Postprocessing
    # class_logits: (total_proposals,(C+1))
    # box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    # proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN
    #                                                 [x1,y1,x2,y2] format)
    class_logits = torch.tensor([[0.1, 0.2, 0.6, 0.1],
                                 [0.4, 0.2, 0.2, 0.2]])
    box_regression = torch.ones(2, 12) * 0.5
    proposals = [torch.tensor([[100, 100, 400, 400]]),
                 torch.tensor([[200, 200, 600, 600]])]

    boxes, scores, labels = net.postprocess_detections(class_logits, box_regression, proposals)
    print(boxes)
    print(scores)
    print(labels)
