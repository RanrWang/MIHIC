import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

from utils.classification import Classification
from utils.utils_metrics import evaluteTop1_5


def eval(save_dir,classes_path,backbone,test_annotation_path,input_shape,device):
    test_annotation_path    = test_annotation_path
    metrics_out_path        = os.path.join(save_dir,'eval')
    _defaults = {
        "model_path"        : os.path.join(save_dir,'best_epoch_weights.pth'),
        "classes_path"      : classes_path,
        "backbone"          : backbone,
        "input_shape"       : input_shape,
    }

    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
    
    classfication = Classification(_defaults)
    
    with open(test_annotation_path,"r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
    with open(os.path.join(metrics_out_path, 'acc_recall.txt'), 'w') as f:
        f.write("top-1 accuracy = %.2f%%\n" % (top1*100))
        f.write("top-5 accuracy = %.2f%%\n" % (top5*100))
        f.write("mean Recall = %.2f%%\n" % (np.mean(Recall)*100))
        f.write("mean Precision = %.2f%%" % (np.mean(Precision)*100))
    
    with open(os.path.join(metrics_out_path,'confusion_matrix.csv'), 'r') as f:
        matrix = f.readlines()
    matrix = [c.strip().split(',') for c in matrix]
    label = matrix[0][1:]
    matrix = np.mat([c[1:] for c in matrix[1:]]).astype(np.int32)
    df = pd.DataFrame(matrix, index=label, columns=label)

    sns.heatmap(data=df,annot=True,fmt="d",cmap="RdBu_r",center=300)
    plt.savefig(os.path.join(metrics_out_path,'confusion_matrix.png'))