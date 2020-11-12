import torch
import torch.nn.functional as F
from torch import nn
import torchvision


class BoxHead(torch.nn.Module):
    def __init__(self, device='cuda', Classes=3, P=7):
        super(BoxHead, self).__init__()

        self.device = device
        self.C = Classes
        self.P = P
        # TODO initialize BoxHead

    def create_ground_truth(self, proposals, gt_labels, bbox):
        """
        This function assigns to each proposal either a ground truth box or the background class
        (we assume background class is 0)
        Input:
        -----
            proposals: list: len(bz){(per_image_proposals, 4)}([x1, y1, x2, y2] format)
            gt_labels: list: len(bz) {(n_obj)}
            bbox: list: len(bz){(n_obj, 4)}
        Output:
        -----
            (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals, 1)(the class that the proposal is assigned)
            regressor_target: (total_proposals, 4)(target encoded in the[t_x, t_y, t_w, t_h] format)
        """

        return labels, regressor_target

    def MultiScaleRoiAlign(self, fpn_feat_list: list, proposals: list, P=7) -> torch.Tensor:
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
                roi_out = torchvision.ops.roi_align(fpn_feat_list[item][batch_i].unsqueeze(0),
                                                    [proposals[batch_i][i].unsqueeze(0)],
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

    def compute_loss(self, class_logits, box_preds, labels, regression_targets,
                     lambda_coeff=1,
                     effective_batch=150):
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

        return loss, loss_class, loss_regr

    def forward(self, feature_vectors):
        """
        Forward the pooled feature vectors through the intermediate layer and the classifier, 
        regressor of the box head
        Input:
        -----
               feature_vectors: (total_proposals, 256*P*P)
        Outputs:
        -----
               class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus 
                                                        background, notice if you want to use
                                                      CrossEntropyLoss you should not pass the 
                                                        output through softmax here)
               box_pred:     (total_proposals,4*C)
        """

        return class_logits, box_pred


if __name__ == '__main__':
    pass
