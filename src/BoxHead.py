import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from utils import matrix_IOU_corner, corner_to_center, center_to_corner


class BoxHead(torch.nn.Module):
    def __init__(self, device='cuda', Classes=3, P=7):
        super(BoxHead, self).__init__()

        self.device = device
        self.C = Classes
        self.P = P

        # initialize BoxHead
        self.intermediate = nn.Sequential(nn.Linear(in_features=256 * P * P, out_features=1024),
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

        feature_vectors = []

        for batch_i in range(len(proposals)):
            w = proposals[batch_i][..., 2] - proposals[batch_i][..., 0]
            h = proposals[batch_i][..., 3] - proposals[batch_i][..., 1]

            # Given a proposal box with w and h we determine the FPN P_k according to the following:
            #   k = floor(4+log_2(sqrt(w*h)/224))
            #   where k is ranging from 2 to 5
            #
            # Get P[k-2] according to the calculated k value.
            #
            k = torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224)).type(torch.int) - 2

            for i, item in enumerate(k):
                # get ROI align
                featmap_size = fpn_feat_list[item][batch_i].shape[-2:]
                bbox = proposals[batch_i][i]
                # rescale bbox region to featmap size
                x_idx = torch.Tensor([0, 2]).type(torch.long)
                y_idx = torch.Tensor([1, 3]).type(torch.long)
                bbox[x_idx] = bbox[x_idx] * featmap_size[0] / img_size[0]
                bbox[y_idx] = bbox[y_idx] * featmap_size[1] / img_size[1]

                roi_out = torchvision.ops.roi_align(fpn_feat_list[item][batch_i].unsqueeze(0),
                                                    [bbox.unsqueeze(0)],
                                                    output_size=(P, P))
                feature_vectors.append(torch.squeeze(roi_out))

        feature_vectors = torch.stack(feature_vectors)
        return torch.reshape(feature_vectors, (feature_vectors.shape[0], -1))

    def postprocess_detections(self, class_logits, box_regression, proposals,
                               conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
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

        return boxes, scores, labels

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

        loss_regr = 0
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
                     random_permutation_foreground=None,
                     random_permutation_background=None) -> tuple:
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

        class_minibatch = torch.Tensor([]).to(self.device)
        class_minibatch_gt = torch.tensor([]).to(self.device)
        reg_minibatch = torch.Tensor([]).to(self.device)
        reg_minibatch_gt = torch.Tensor([]).to(self.device)

        no_bg_idx = (labels > 0).nonzero(as_tuple=False).squeeze()
        bg_idx = (labels == 0).nonzero(as_tuple=False).squeeze()

        if random_permutation_foreground is None and random_permutation_background is None:
            M_no_bg = int(effective_batch / 4)
            M_bg = effective_batch - M_no_bg

            if torch.sum(labels != 0) > M_no_bg:
                # Randomly choose M/2 anchors with positive ground truth labels.
                rand_idx = torch.randint(no_bg_idx.size()[0], (M_no_bg,))
                class_minibatch = class_logits[no_bg_idx[rand_idx]]
                class_minibatch_gt = labels[no_bg_idx[rand_idx]]
                reg_minibatch = box_preds[no_bg_idx[rand_idx]]
                reg_minibatch_gt = regression_targets[no_bg_idx[rand_idx]]
            else:
                class_minibatch = class_logits[no_bg_idx]
                class_minibatch_gt = labels[no_bg_idx]
                reg_minibatch = box_preds[no_bg_idx]
                reg_minibatch_gt = regression_targets[no_bg_idx]

            if torch.sum(labels == 0) > M_bg:
                rand_idx = torch.randint(bg_idx.size()[0], (M_bg,))
                class_minibatch = torch.cat(
                    [class_minibatch, class_logits[bg_idx[rand_idx]]], dim=0)
                class_minibatch_gt = torch.cat(
                    [class_minibatch_gt, labels[bg_idx[rand_idx]]], dim=0)
                reg_minibatch = torch.cat([reg_minibatch, box_preds[bg_idx[rand_idx]]], dim=0)
                reg_minibatch_gt = torch.cat(
                    [reg_minibatch_gt, regression_targets[bg_idx[rand_idx]]], dim=0)
            else:
                class_minibatch = torch.cat([class_minibatch, class_logits[bg_idx]], dim=0)
                class_minibatch_gt = torch.cat([class_minibatch_gt, labels[bg_idx]], dim=0)
                reg_minibatch = torch.cat([reg_minibatch, box_preds[bg_idx]], dim=0)
                reg_minibatch_gt = torch.cat([reg_minibatch_gt, regression_targets[bg_idx]], dim=0)

        else:
            class_minibatch = class_logits[no_bg_idx][random_permutation_foreground]
            class_minibatch_gt = labels[no_bg_idx][random_permutation_foreground]
            reg_minibatch = box_preds[no_bg_idx][random_permutation_foreground]
            reg_minibatch_gt = regression_targets[no_bg_idx][random_permutation_foreground]

            class_minibatch = torch.cat(
                [class_minibatch, class_logits[bg_idx][random_permutation_background]], dim=0)
            class_minibatch_gt = torch.cat(
                [class_minibatch_gt, labels[bg_idx][random_permutation_background]], dim=0)
            reg_minibatch = torch.cat(
                [reg_minibatch, box_preds[bg_idx][random_permutation_background]], dim=0)
            reg_minibatch_gt = torch.cat(
                [reg_minibatch_gt, regression_targets[bg_idx][random_permutation_background]], dim=0)

        class_minibatch_gt = class_minibatch_gt.type(torch.long)

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


import os
if __name__ == '__main__':
    net = BoxHead(device='cuda', Classes=3, P=7)

    # Test Ground Truth creation
    # testcase 7 has one incorrect label because the iou of that one is 0.4999
    # TODO: testcase 3 has many incorrect regressor targets when running on cpu
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
    # TODO: test 2 & 3 still have differences ONLY in the last line output; reasons unknown
    roi_dir = "test/MultiScaleRoiAlign/"
    num_test = 0
    # load test cases
    path = os.path.join(roi_dir, "multiscale_RoIAlign_test" + str(num_test) + ".pt")
    fpn_feat_list = [item.cuda() for item in torch.load(path)['fpn_feat_list']]
    proposals = [item.cuda() for item in torch.load(path)['proposals']]
    output_feature_vectors = torch.load(path)['output_feature_vectors'].cuda()
    feature_vectors = net.MultiScaleRoiAlign(fpn_feat_list, proposals)
    print("\n----- ROI align test -----")
    print(feature_vectors)
    print(output_feature_vectors)

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
                             random_permutation_foreground=random_permutation_foreground,
                             random_permutation_background=random_permutation_background)

        print("loss_class_gt", loss_class_gt)
        print("loss_class   ", loss_class)
        print("loss_reg_gt  ", loss_reg_gt)
        print("loss_reg     ", loss_reg)
