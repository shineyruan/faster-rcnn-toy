# sys modules
import os
import sys
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# we will need the ImageList from torchvision
from torchvision.models.detection.image_list import ImageList


# local modules
from dataset import BuildDataLoader, BuildDataset
from pretrained_models import pretrained_models_680
from BoxHead import BoxHead

# env variables
IN_COLAB = 'google' in sys.modules
COLAB_ROOT = "/content/drive/My Drive/CIS680_2019/Faster-RCNN"


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
    checkpoints_path = 'checkpoints/'
    if IN_COLAB:
        imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        masks_path = os.path.join(COLAB_ROOT, masks_path)
        labels_path = os.path.join(COLAB_ROOT, labels_path)
        bboxes_path = os.path.join(COLAB_ROOT, bboxes_path)
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)

    os.makedirs(checkpoints_path, exist_ok=True)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
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
    keep_topK = 200

    # Initialize box head DL network
    box_head = BoxHead(device=net_device).to(net_device)
    box_head.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(box_head.parameters(), lr=0.0007)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 13], gamma=0.1)
    num_epochs = 40

    # global loss list
    list_avg_train_loss = []
    list_class_train_loss = []
    list_regression_train_loss = []
    list_avg_test_loss = []
    list_class_test_loss = []
    list_regression_test_loss = []

    for epoch in range(num_epochs):
        count = 0
        avg_train_loss = 0
        avg_class_train_loss = 0
        avg_regression_train_loss = 0

        # initialize tqdm
        enumerate_tqdm = tqdm(enumerate(train_loader, 0))
        box_head.train()
        for iter, batch in enumerate_tqdm:
            images, labels, _, bbox, index = batch
            images = images.to(net_device)

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

            feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals)
            class_logits, box_preds = box_head.forward(feature_vectors)
            labels, regressor_target = box_head.create_ground_truth(proposals,
                                                                    labels,
                                                                    bbox)
            loss, loss_class, loss_reg = box_head.compute_loss(class_logits,
                                                               box_preds,
                                                               labels.to(net_device),
                                                               regressor_target.to(net_device),
                                                               effective_batch=32)
            loss.backward()
            optimizer.step()

            enumerate_tqdm.set_description("training loss = %f" % loss.item())

            avg_train_loss += loss.item()
            avg_class_train_loss += loss_class.item()
            avg_regression_train_loss += loss_reg.item()
            count += 1

        avg_train_loss /= count
        avg_class_train_loss /= count
        avg_regression_train_loss /= count

        list_avg_train_loss.append(avg_train_loss)
        list_class_train_loss.append(avg_class_train_loss)
        list_regression_train_loss.append(avg_regression_train_loss)

        # eval mode
        count = 0
        avg_test_loss = 0
        avg_class_test_loss = 0
        avg_regression_test_loss = 0

        enumerate_tqdm = tqdm(enumerate(test_loader, 0))
        box_head.eval()
        with torch.no_grad():
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
                class_logits, box_preds = box_head.forward(feature_vectors)
                labels, regressor_target = box_head.create_ground_truth(proposals,
                                                                        labels,
                                                                        bbox)
                loss, loss_class, loss_reg = box_head.compute_loss(class_logits,
                                                                   box_preds,
                                                                   labels.to(net_device),
                                                                   regressor_target.to(net_device),
                                                                   effective_batch=32)

                avg_test_loss += loss.item()
                avg_class_test_loss += loss_class.item()
                avg_regression_test_loss += loss_reg.item()
                count += 1

        avg_test_loss /= count
        avg_class_test_loss /= count
        avg_regression_test_loss /= count

        print("Epoch: {}, loss={}, class loss={}, regression loss={}".format(
            epoch,
            avg_train_loss,
            avg_class_train_loss,
            avg_regression_train_loss
        ))

        list_avg_test_loss.append(avg_test_loss)
        list_class_test_loss.append(avg_class_test_loss)
        list_regression_test_loss.append(avg_regression_test_loss)

        path = os.path.join(checkpoints_path, 'rpn_epoch' + str(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': box_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_running_loss': avg_train_loss,
            'train_class_loss': avg_class_train_loss,
            'train_regression_loss': avg_regression_train_loss,
            'list_train_loss': list_avg_train_loss,
            'list_class_train_loss': list_class_train_loss,
            'list_regression_train_loss': list_regression_train_loss,
            'test_running_loss': avg_test_loss,
            'test_class_loss': avg_class_test_loss,
            'test_regression_loss': avg_regression_test_loss,
            'list_test_loss': list_avg_test_loss,
            'list_class_test_loss': list_class_test_loss,
            'list_regression_test_loss': list_regression_test_loss
        }, path)
