# sys modules
import os
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# local modules
from dataset import BuildDataLoader, BuildDataset
from pretrained_models import pretrained_models_680
from BoxHead import BoxHead


if __name__ == "__main__":
    boxhead_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    # Put the path were you save the given pretrained model
    pretrained_path = 'model/checkpoint680.pth'
    backbone, rpn = pretrained_models_680(pretrained_path, device=backbone_device)

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Standard Dataloaders Initialization
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 8
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

    # Initialize box head DL network
    box_head = BoxHead(device=boxhead_device).to(boxhead_device)
    box_head.train()

    # initialize tqdm
    enumerate_tqdm = tqdm(enumerate(train_loader, 0))
    for iter, batch in enumerate_tqdm:
        images, *other = batch
        images = images.to(backbone_device)

        with torch.no_grad():
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

        for item in fpn_feat_list:
            item = item.to(boxhead_device)
        feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals)

        class_out, reg_out = box_head.forward(feature_vectors)
