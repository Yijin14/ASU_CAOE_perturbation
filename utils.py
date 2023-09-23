# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:52:33 2019

@author: Keshik
"""
import os
import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import pandas as pd
from PIL import Image

# object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
#                      'bottle', 'bus', 'car', 'cat', 'chair',
#                      'cow', 'diningtable', 'dog', 'horse',
#                      'motorbike', 'person', 'pottedplant',
#                      'sheep', 'sofa', 'train', 'tvmonitor']

animal = ['bird', 'dog']#['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
vehicle = ['aeroplane', 'car']#['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train']
object_categories = animal + vehicle

def get_categories(labels_dir):
    """
    Get the object categories
    
    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """
    
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError
    
    else:
        categories = []
        
        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])
        
        return categories


def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding 
    
    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """
    
    ls = target['annotation']['object']
    bbox = target['annotation']['object'][0]['bndbox']
  
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
  
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    
    k = np.zeros(len(object_categories))
    k[j] = 1
    label = torch.from_numpy(k)
    bilabel = torch.from_numpy(np.array([1,0])) if ls[0]['name'] in animal else torch.from_numpy(np.array([0,1]))
  
    return {'label':label, 'bilabel':bilabel, 'bbox':bbox}#torch.from_numpy(k)


def get_nrows(file_name):
    """
    Get the number of rows of a csv file
    
    Args:
        file_path: path of the csv file
    Raises:
        FileNotFoundError: If the csv file does not exist
    Returns:
        number of rows
    """
    
    if not os.path.isfile(file_name):
        raise FileNotFoundError
    
    s = 0
    with open(file_name) as f:
        s = sum(1 for line in f)
    return s


def get_mean_and_std(dataloader):
    """
    Get the mean and std of a 3-channel image dataset 
    
    Args:
        dataloader: pytorch dataloader
    Returns:
        mean and std of the dataset
    """
    mean = []
    std = []
    
    total = 0
    r_running, g_running, b_running = 0, 0, 0
    r2_running, g2_running, b2_running = 0, 0, 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            r, g, b = data[:,0 ,:, :], data[:, 1, :, :], data[:, 2, :, :]
            r2, g2, b2 = r**2, g**2, b**2
            
            # Sum up values to find mean
            r_running += r.sum().item()
            g_running += g.sum().item()
            b_running += b.sum().item()
            
            # Sum up squared values to find standard deviation
            r2_running += r2.sum().item()
            g2_running += g2.sum().item()
            b2_running += b2.sum().item()
            
            total += data.size(0)*data.size(2)*data.size(3)
    
    # Append the mean values 
    mean.extend([r_running/total, 
                 g_running/total, 
                 b_running/total])
    
    # Calculate standard deviation and append
    std.extend([
            math.sqrt((r2_running/total) - mean[0]**2),
            math.sqrt((g2_running/total) - mean[1]**2),
            math.sqrt((b2_running/total) - mean[2]**2)
            ])
    
    return mean, std


def plot_history(hist_list, label_list, y_label, filename, c_list=[], labels=["train", "validation"]):
    """
    Plot training and validation history
    
    Args:
        train_hist: numpy array consisting of train history values (loss/ accuracy metrics)
        valid_hist: numpy array consisting of validation history values (loss/ accuracy metrics)
        y_label: label for y_axis
        filename: filename to store the resulting plot
        labels: legend for the plot
        
    Returns:
        None
    """
    # Plot loss and accuracy
    xi = [i for i in range(0, len(hist_list[0]), 2)]
    if c_list==[]:
        c_list = ['k']*len(hist_list)
    plt.clf()
    # plt.plot(train_hist, label = labels[0])
    # plt.plot(val_hist, label = labels[1])
    for hist,label,c in zip(hist_list,label_list,c_list):
        plt.plot(hist,c, label = label)
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()


def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0
    
    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
    
    return scores

def save_results(images, scores, columns, filename):
    """
    Save inference results as csv
    
    Args:
        images: inferred image list
        scores: confidence score for inferred images
        columns: object categories
        filename: name and location to save resulting csv
    """
    df_scores = pd.DataFrame(scores, columns=columns)
    df_scores['image'] = images
    df_scores.set_index('image', inplace=True)
    df_scores.to_csv(filename)


def append_gt(gt_csv_path, scores_csv_path, store_filename):
    """
    Append ground truth to confidence score csv
    
    Args:
        gt_csv_path: Ground truth csv location
        scores_csv_path: Confidence scores csv path
        store_filename: name and location to save resulting csv
    """
    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)
    
    gt_label_list = []
    for index, row in gt_df.iterrows():
        arr = np.array(gt_df.iloc[index,1:], dtype=int)
        target_idx = np.ravel(np.where(arr == 1))
        j = [object_categories[i] for i in target_idx]
        gt_label_list.append(j)
    
    scores_df.insert(1, "gt", gt_label_list)
    scores_df.to_csv(store_filename, index=False)

        

def get_classification_accuracy(gt_csv_path, scores_csv_path, store_filename):
    """
    Plot mean tail accuracy across all classes for threshold values
    
    Args:
        gt_csv_path: Ground truth csv location
        scores_csv_path: Confidence scores csv path
        store_filename: name and location to save resulting plot
    """
    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)
    
    # Get the top-50 images
    top_num = 2800
    image_num = 2
    num_threshold = 10
    results = []
    
    for image_num in range(1, 21):
        clf = np.sort(np.array(scores_df.iloc[:,image_num], dtype=float))[-top_num:]
        ls = np.linspace(0.0, 1.0, num=num_threshold)
        
        class_results = []
        for i in ls:
            clf = np.sort(np.array(scores_df.iloc[:,image_num], dtype=float))[-top_num:]
            clf_ind = np.argsort(np.array(scores_df.iloc[:,image_num], dtype=float))[-top_num:]
            
            # Read ground truth
            gt = np.sort(np.array(gt_df.iloc[:,image_num], dtype=int))
            
            # Now get the ground truth corresponding to top-50 scores
            gt = gt[clf_ind]
            clf[clf >= i] = 1
            clf[clf < i] = 0
            
            score = accuracy_score(y_true=gt, y_pred=clf, normalize=False)/clf.shape[0]
            class_results.append(score)
        
        results.append(class_results)
    
    results = np.asarray(results)
    
    ls = np.linspace(0.0, 1.0, num=num_threshold)
    plt.plot(ls, results.mean(0))
    plt.title("Mean Tail Accuracy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Mean Tail Accuracy")
    plt.savefig(store_filename)
    plt.show()
            

#get_classification_accuracy("../models/resnet18/results.csv", "../models/resnet18/gt.csv", "roc-curve.png")


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    print([data, target])
    return [data, target]

def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return (imgs, torch.FloatTensor(annot_padded))


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample[0], sample[1]
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class my_Padder(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, color=(0,0,0)):
        self.color = color

    def __call__(self, sample, common_size=500):
        image = sample
        height, width = image.size
        # pix = np.array(image.getdata())#.reshape(height, width, 3)

        # new_image = np.zeros((common_size, common_size, 3))
        # new_image[0:height, 0:width] = pix

        # print(f'{pix.shape}\n{new_image.shape}')
        # exit()

        # new_image = Image.fromarray(new_image.astype('uint8'))#.convert('RGB')

        new_image = Image.new(image.mode, (common_size, common_size), self.color)
        new_image.paste(image, (0, 0))
        # new_image.save('visualize.jpg')
        # print(new_image.mode)
        # exit()

        return new_image


def visualize(classifier2, classifier10, generator, test_loader, num=3):
    device = 'cpu'
    map = {1:'vehicle', 0:'animal'}
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print('>>data shape: ', example_data.shape)
    example_data=example_data.to(device)
    bbox = example_targets['bbox']
    mask = example_targets['mask']
    example_targets=example_targets['label'].to(device)
    # print(example_data.cpu()[0].shape)
    with torch.no_grad():
        attacked_data = generator(example_data,bbox,mask)
        output_old = classifier10(example_data)
        output_new = classifier10(attacked_data)
        parity_old = classifier2(example_data)
        parity_new = classifier2(attacked_data)
    fig = plt.figure()
    for index in range(num):
        i=index*2+1
        print(output_old.shape)
        label_old = output_old.data.max(1, keepdim=True)[1][i].item()
        label_new = output_new.data.max(1, keepdim=True)[1][i].item()
        plt.subplot(2,num,index+1)
        plt.tight_layout()
        plt.imshow(example_data.cpu()[i].permute(1, 2, 0) , interpolation='none')
        xs = [int(item) for item in [bbox['xmin'][i],bbox['xmax'][i],bbox['xmax'][i],bbox['xmin'][i],bbox['xmin'][i]]]
        ys = [int(item) for item in [bbox['ymin'][i],bbox['ymin'][i],bbox['ymax'][i],bbox['ymax'][i],bbox['ymin'][i]]]
        print(f'{xs},{ys}')
        plt.plot(xs, ys, color="red")#gca().add_patch(Rectangle((bbox['xmin'][i],bbox['ymin'][i]),bbox['xmax'][i]-bbox['xmin'][i],bbox['ymax'][i]-bbox['ymin'][i],linewidth=1,edgecolor='r',facecolor='none'))#plot(xs, ys, color="red")
        plt.title(f"{map[parity_old.data.max(1, keepdim=True)[1][i].item()]}: {object_categories[label_old]}")
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.subplot(2,num,index+1+num)
        # plt.tight_layout()
        # # st()
        # # plt.imshow(torch.stack((attacked_data.cpu()[i][0]-example_data.cpu()[i][0], example_data.cpu()[i][0]-attacked_data.cpu()[i][0], torch.zeros_like(attacked_data.cpu()[i][0])),2), cmap='bwr', interpolation='none')
        # plt.title("")
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(3,2,i+5)
        plt.tight_layout()
        plt.imshow(attacked_data.cpu()[i].permute(1, 2, 0) , interpolation='none')
        plt.text(263, 0,f"{map[parity_new.data.max(1, keepdim=True)[1][i].item()]}: ",color = 'green' if parity_new.data.max(1, keepdim=True)[1][i].item()==parity_old.data.max(1, keepdim=True)[1][i].item() else 'red',ha ='right', va="bottom")
        plt.text(263.90, 0,f"{object_categories[label_new]}",color = 'green' if label_new==label_old else 'red', va="bottom")
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.savefig(f'visualize.png',bbox_inches='tight')