# sys module
from matplotlib import pyplot as plt
import torch

# local module
from main_train import os, COLAB_ROOT, IN_COLAB


if __name__ == "__main__":
    figures_path = 'figures/'
    checkpoints_path = 'checkpoint_save/'
    if IN_COLAB:
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)
        figures_path = os.path.join(COLAB_ROOT, figures_path)

    os.makedirs(figures_path, exist_ok=True)

    epoch = 39
    path = os.path.join(checkpoints_path, "rpn_epoch" + str(epoch))
    checkpoint = torch.load(path)

    list_train_loss = checkpoint['list_train_loss']
    list_class_train_loss = checkpoint['list_class_train_loss']
    list_regression_train_loss = checkpoint['list_regression_train_loss']
    list_test_loss = checkpoint['list_test_loss']
    list_class_test_loss = checkpoint['list_class_test_loss']
    list_regression_test_loss = checkpoint['list_regression_test_loss']

    plt.figure()
    plt.plot(list_train_loss)
    plt.title('Training Total Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    path = os.path.join(figures_path, 'training_loss.png')
    plt.savefig(path)

    plt.figure()
    plt.plot(list_test_loss)
    plt.title('Testing Total Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    path = os.path.join(figures_path, 'testing_loss.png')
    plt.savefig(path)

    plt.figure()
    plt.plot(list_class_train_loss)
    plt.title('Training Classification Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    path = os.path.join(figures_path, 'training_class_loss.png')
    plt.savefig(path)

    plt.figure()
    plt.plot(list_regression_train_loss)
    plt.title('Training Regression Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    path = os.path.join(figures_path, 'training_regress_loss.png')
    plt.savefig(path)

    plt.figure()
    plt.plot(list_class_test_loss)
    plt.title('Testing Classification Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    path = os.path.join(figures_path, 'testing_class_loss.png')
    plt.savefig(path)

    plt.figure()
    plt.plot(list_regression_test_loss)
    plt.title('Testing Regression Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    path = os.path.join(figures_path, 'testing_regress_loss.png')
    plt.savefig(path)
