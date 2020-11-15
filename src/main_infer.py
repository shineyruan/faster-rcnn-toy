# sys modules
import torch
import sys
import os
from tqdm import tqdm
import numpy as np
from torchvision.models.detection.image_list import ImageList
from matplotlib import pyplot as plt

# local modules
from dataset import BuildDataset, BuildDataLoader
from BoxHead import BoxHead
from utils import visual_bbox_mask, cv2
from pretrained_models import pretrained_models_680

# env variables
IN_COLAB = 'google' in sys.modules
COLAB_ROOT = "/content/drive/My Drive/CIS680_2019/Mask-RCNN"

if __name__ == "__main__":
    net_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    # Put the path were you save the given pretrained model
    pretrained_path = 'model/checkpoint680.pth'
    backbone, rpn = pretrained_models_680(pretrained_path, device=net_device)

    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = 'data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = 'data/hw3_mycocodata_bboxes_comp_zlib.npy'
    checkpoints_path = 'checkpoint_save/'
    images_path = 'out_images/'
    mAP_path = "mAP/"
    figures_path = 'figures/'
    if IN_COLAB:
        imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        masks_path = os.path.join(COLAB_ROOT, masks_path)
        labels_path = os.path.join(COLAB_ROOT, labels_path)
        bboxes_path = os.path.join(COLAB_ROOT, bboxes_path)
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)
        images_path = os.path.join(COLAB_ROOT, images_path)
        mAP_path = os.path.join(COLAB_ROOT, mAP_path)
        figures_path = os.path.join(COLAB_ROOT, figures_path)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Standard Dataloaders Initialization
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 2  # on RTX 2060 Super with 8GB RAM, maximum batch size is 7
    print("batch size:", batch_size)
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()

    # Here we keep the top 20, but during training you
    #   should keep around 200 boxes from the 1000 proposals
    keep_topK = 20

    # load checkpoint
    checkpoint_list = ["rpn_epoch39-1", "rpn_epoch49-2"]
    path = os.path.join(checkpoints_path, checkpoint_list[-1])
    checkpoint = torch.load(path)

    # create device and RPN net
    box_head = BoxHead(device=net_device).to(net_device)
    box_head.load_state_dict(checkpoint['model_state_dict'])
    box_head.eval()

    trues_per_batch = []
    positives_per_batch = []
    match_values = []
    score_values = []

    with torch.no_grad():
        # loop over all images
        enumerate_tqdm = tqdm(enumerate(test_loader, 0))
        for iter, batch in enumerate_tqdm:
            images, labels, _, bbox, index = batch
            images = images.to(net_device)
            labels = [item.to(net_device) for item in labels]
            bbox = [item.to(net_device) for item in bbox]

            # Take the features from the backbone
            backbone_out = backbone(images)

            # The RPN implementation takes as first argument the following image list
            im_list = ImageList(images, [(800, 1088)] * images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpn_out = rpn(im_list, backbone_out)

            # The final output is
            # A list of proposal tensors:
            #   list:len(bz){(keep_topK,4)}
            proposals = [proposal[0:keep_topK, :] for proposal in rpn_out[0]]
            # A list of features produces by the backbone's FPN levels:
            #   list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list = list(backbone_out.values())

            feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals)
            class_logits, box_preds = box_head.forward(feature_vectors, training=False)
            nms_boxes, nms_scores, nms_labels = box_head.postprocess_detections(
                class_logits, box_preds, proposals, keep_num_postNMS=2)
            top_boxes, _, top_labels = box_head.get_top_K(
                class_logits, box_preds, proposals)

            matches, scores, num_trues, num_positives = \
                box_head.box_head_evaluation(nms_boxes, nms_scores, nms_labels,
                                             bbox, labels)

            trues_per_batch.append(num_trues)
            positives_per_batch.append(num_positives)
            match_values.append(matches)
            score_values.append(scores)

            for i in range(images.shape[0]):
                out_img = visual_bbox_mask(images[i].cpu(), top_boxes[i].cpu(),
                                           top_labels[i].cpu())

                image_path = os.path.join(images_path, 'visual_output_' +
                                          str(iter) + '_' + str(i) + 'top_K.png')
                cv2.imwrite(image_path, out_img)
                cv2.imshow("visualize output", out_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                out_img = visual_bbox_mask(images[i].cpu(), nms_boxes[i].cpu(),
                                           nms_labels[i].cpu(), nms_scores[i].cpu())

                image_path = os.path.join(images_path, 'visual_output_' +
                                          str(iter) + '_' + str(i) + 'after_nms.png')
                cv2.imwrite(image_path, out_img)
                cv2.imshow("visualize output", out_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    trues_per_batch = torch.stack(trues_per_batch)
    positives_per_batch = torch.stack(positives_per_batch)
    trues_per_batch = torch.sum(trues_per_batch, dim=0)
    positives_per_batch = torch.sum(positives_per_batch, dim=0)
    match_values = torch.cat(match_values)
    score_values = torch.cat(score_values)

    os.makedirs(mAP_path, exist_ok=True)
    path = os.path.join(mAP_path, 'matches.pt')
    torch.save({
        'trues per batch': trues_per_batch,
        'positives per batch': positives_per_batch,
        'match_values': match_values,
        'score_values': score_values
    }, path)

    # calculate mAP
    list_sorted_recall = []
    list_sorted_precision = []
    list_AP = []

    AP = 0
    cnt = 0
    for class_i in range(3):
        if torch.sum(match_values[:, class_i]) > 0:
            area, sorted_recall, sorted_precision = \
                box_head.average_precision(match_values[:, class_i],
                                           score_values[:, class_i],
                                           trues_per_batch[class_i],
                                           positives_per_batch[class_i],
                                           threshold=0.5)
            AP += area
            cnt += 1

            list_sorted_recall.append(sorted_recall)
            list_sorted_precision.append(sorted_precision)
            list_AP.append(area)

    mAP = AP if cnt == 0 else AP / cnt
    # calculate mean loss
    print('testing mAP   {}'.format(mAP))

    path = os.path.join(mAP_path, 'mAP')
    torch.save({
        'sorted_recalls': list_sorted_recall,
        'sorted_precisions': list_sorted_precision,
        'AP': list_AP,
        'mAP': mAP
    }, path)

    for i in range(len(list_sorted_recall)):
        plt.figure()
        plt.plot(list_sorted_recall[i], list_sorted_precision[i], '.-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(figures_path + "class_" + str(i) + ".png")
