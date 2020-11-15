import torch
import os
from matplotlib import pyplot as plt

from main_train import IN_COLAB, COLAB_ROOT


if __name__ == "__main__":
    mAP_path = "mAP/"
    figure_path = "figures/"

    if IN_COLAB:
        mAP_path = os.path.join(COLAB_ROOT, mAP_path)
        figure_path = os.path.join(COLAB_ROOT, figure_path)

    path = os.path.join(mAP_path, 'mAP')
    list_sorted_recall = torch.load(path)['sorted_recalls']
    list_sorted_precision = torch.load(path)['sorted_precisions']
    list_AP = torch.load(path)['AP']

    print(list_sorted_recall)
    print(list_sorted_precision)

    os.makedirs(figure_path, exist_ok=True)

    for i, sorted_recall in enumerate(list_sorted_recall):
        plt.figure()
        plt.plot(sorted_recall, list_sorted_precision[i], '.-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs. Recall Curve for Class {}'.format(i))
        plt.savefig(figure_path + "class_" + str(i) + ".png")
