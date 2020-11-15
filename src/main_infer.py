# sys modules
import torch
import sys
import os
from tqdm import tqdm
import numpy as np
from torchvision.models.detection.image_list import ImageList

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
    if IN_COLAB:
        imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        masks_path = os.path.join(COLAB_ROOT, masks_path)
        labels_path = os.path.join(COLAB_ROOT, labels_path)
        bboxes_path = os.path.join(COLAB_ROOT, bboxes_path)
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)
        images_path = os.path.join(COLAB_ROOT, images_path)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    os.makedirs(images_path, exist_ok=True)

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

    # save test figures

    with torch.no_grad():
        # loop over all images
        enumerate_tqdm = tqdm(enumerate(test_loader, 0))
        for iter, batch in enumerate_tqdm:
            images, labels, _, bbox, index = batch
            images = images.to(net_device)

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

            matches, scores, num_trues, num_positives = \
                box_head.box_head_evaluation(nms_boxes, nms_scores, nms_labels,
                                             bbox, labels)

            for i in range(images.shape[0]):
                out_img = visual_bbox_mask(images[i].cpu(), nms_boxes[i].cpu(),
                                           nms_scores[i].cpu(), nms_labels[i].cpu())

                image_path = os.path.join(images_path, 'visual_output_' +
                                          str(iter) + '_' + str(i) + 'after_nms.png')
                cv2.imwrite(image_path, out_img)
                cv2.imshow("visualize output", out_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
