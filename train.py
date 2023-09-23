# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:37:39 2019

@author: Keshik
"""

from tqdm import tqdm
import torch
import gc
import os
from utils import get_ap_score
import numpy as np
from pdb import set_trace as st
from utils import encode_labels, plot_history

def train_classifier(num_class, model, generator, device, optimizer, scheduler, train_loader, valid_loader, save_dir, model_num, epochs, log_file):
    """
    Train a deep neural network model
    
    Args:
        model: pytorch model object
        device: cuda or cpu
        optimizer: pytorch optimizer object
        scheduler: learning rate scheduler object that wraps the optimizer
        train_dataloader: training  images dataloader
        valid_dataloader: validation images dataloader
        save_dir: Location to save model weights, plots and log_file
        epochs: number of training epochs
        log_file: text file instance to record training and validation history
        
    Returns:
        Training history and Validation history (loss and average precision)
    """
    
    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0
    
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        # log_file.write("Epoch {} >>".format(epoch+1))
        scheduler.step()
        
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_ap = 0.0
            
            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            m = torch.nn.Sigmoid()
            
            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                for data, target in tqdm(train_loader):
                    # st()
                    # print(data.size())
                    bbox = target['bbox']
                    mask = target['mask']
                    if num_class == 2:
                        target = target['bilabel']
                    else:
                        target = target['label']
                    target = target.float()
                    data, target, mask = data.to(device), target.to(device), mask.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    if generator != None:
                        data = generator(data, 'bbox', mask)
                    output = model(data)
                    
                    loss = criterion(output, target)
                    
                    # Get metrics here
                    running_loss += loss # sum up batch loss
                    running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 
               
                    # Backpropagate the system the determine the gradients
                    loss.backward()
                    
                    # Update the paramteres of the model
                    optimizer.step()
            
                    # clear variables
                    del data, target, output
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    #print("loss = ", running_loss)
                    
                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_map_ = running_ap/num_samples
                
                print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
                    tr_loss_, tr_map_))
                
                # log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
                #     tr_loss_, tr_map_))
                
                # Append the values to global arrays
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                        
                        
            else:
                model.train(False)  # Set model to evaluate mode
        
                # torch.no_grad is for memory savings
                with torch.no_grad():
                    for data, target in tqdm(valid_loader):
                        # bbox = target['bbox']
                        # mask = target['mask']
                        if num_class == 2:
                            target = target['bilabel']
                        else:
                            target = target['label']
                        target = target.float()
                        data, target = data.to(device), target.to(device)#, mask.to(device)
                        # if generator != None:
                        #     data = generator(data, bbox, mask)
                        output = model(data)
                        
                        loss = criterion(output, target)
                        
                        running_loss += loss # sum up batch loss
                        running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 
                        
                        del data, target, output
                        gc.collect()
                        torch.cuda.empty_cache()

                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item()/num_samples
                    val_map_ = running_ap/num_samples
                    
                    # Append the values to global arrays
                    val_loss.append(val_loss_), val_map.append(val_map_)
                
                    print('val_loss: {:.4f}, val_avg_precision:{:.3f}'.format(
                    val_loss_, val_map_))
                    
                    # log_file.write('val_loss: {:.4f}, val_avg_precision:{:.3f}\n'.format(
                    # val_loss_, val_map_))
                    
                    # Save model using val_acc
                    if val_map_ >= best_val_map:
                        best_val_map = val_map_
                        # log_file.write("saving best weights...\n")
                        # torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))
                    
    return ([tr_loss, tr_map], [val_loss, val_map])

    

def test_classifier(num_class, model, generator, device, test_loader, returnAllScores=False):
    """
    Evaluate a deep neural network model
    
    Args:
        model: pytorch model object
        device: cuda or cpu
        test_dataloader: test images dataloader
        returnAllScores: If true addtionally return all confidence scores and ground truth 
        
    Returns:
        test loss and average precision. If returnAllScores = True, check Args
    """
    model.train(False)
    
    running_loss = 0
    running_ap = 0
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    m = torch.nn.Sigmoid()
    
    if returnAllScores == True:
        all_scores = np.empty((0, 20), float)
        ground_scores = np.empty((0, 20), float)
        
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            # bbox = target['bbox']
            # mask = target['mask']
            if num_class == 2:
                target = target['bilabel']
            else:
                target = target['label']
            #print(data.size(), target.size())
            target = target.float()
            data, target = data.to(device), target.to(device)#, mask.to(device)
            # bs, ncrops, c, h, w = data.size()

            # output = model(data.view(-1, c, h, w))
            # output = output.view(bs, ncrops, -1).mean(1)
            # if generator != None:
            #     data = generator(data, bbox, mask)
            output = model(data)
            
            loss = criterion(output, target)
            
            running_loss += loss # sum up batch loss
            running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 
            
            if returnAllScores == True:
                all_scores = np.append(all_scores, torch.Tensor.cpu(m(output)).detach().numpy() , axis=0)
                ground_scores = np.append(ground_scores, torch.Tensor.cpu(target).detach().numpy() , axis=0)
            
            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item()/num_samples
    test_map = running_ap/num_samples
    
    print('test_loss: {:.4f}, test_avg_precision:{:.3f}'.format(
                    avg_test_loss, test_map))
    
    
    if returnAllScores == False:
        return avg_test_loss, running_ap
    
    return avg_test_loss, running_ap, all_scores, ground_scores



def train_generator(classifier_target, classifier_sensitive, model, device, optimizer, scheduler, train_loader, valid_loader, test_loader, epochs, log_file):
    sen_tr_loss, tar_tr_loss, sen_tr_map, tar_tr_map = [], [], [], []
    sen_val_loss, tar_val_loss, sen_val_map, tar_val_map = [], [], [], []

    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        for phase in ['train', 'valid', 'test']:
            running_loss_target = 0.0
            running_loss_sensitive = 0.0
            running_ap_target = 0.0
            running_ap_sensitive = 0.0

            criterion = torch.nn.CrossEntropyLoss()#torch.nn.BCEWithLogitsLoss(reduction='sum')#
            m = torch.nn.Sigmoid()

            classifier_sensitive.eval()
            classifier_target.eval()

            if phase == 'train':
                model.train(True) 
                for data, target in tqdm(train_loader):
                    bilabel = target['bilabel'].float()
                    label = target['label'].float()
                    bbox = target['bbox']
                    mask = target['mask']
                    # print(bbox)
                    # exit()
                    data, label, bilabel, mask = data.to(device), label.to(device), bilabel.to(device), mask.to(device)

                    optimizer.zero_grad()

                    attacked_data = model(data, 'bbox', mask)

                    output_target = classifier_target(attacked_data)
                    output_sensitive = classifier_sensitive(attacked_data)

                    loss_target = criterion(output_target, bilabel)
                    loss_sensitive = criterion(output_sensitive, label)
                    loss = loss_target-loss_sensitive
                    loss.backward()
                    optimizer.step()
                    # print(f'loss_target{loss_target}\tloss_sensitive{loss_sensitive}')

                    running_loss_target += loss_target # sum up batch loss
                    running_loss_sensitive += loss_sensitive # sum up batch loss
                    running_ap_target += get_ap_score(torch.Tensor.cpu(bilabel).detach().numpy(), torch.Tensor.cpu(m(output_target)).detach().numpy()) 
                    running_ap_sensitive += get_ap_score(torch.Tensor.cpu(label).detach().numpy(), torch.Tensor.cpu(m(output_sensitive)).detach().numpy()) 

                    del data, target, output_target, output_sensitive, bilabel, label, bbox, mask
                    gc.collect()
                    torch.cuda.empty_cache()

                num_samples = float(len(train_loader.dataset))
                tr_loss_target = running_loss_target.item()/num_samples
                tr_loss_sensitive = running_loss_sensitive.item()/num_samples
                tr_map_target = running_ap_target/num_samples
                tr_map_sensitive = running_ap_sensitive/num_samples
                print(f'loss_target{tr_loss_target}\tloss_sensitive{tr_loss_sensitive}')
                print(f'acc_target{tr_map_target}\tacc_sensitive{tr_map_sensitive}')

                sen_tr_loss.append(tr_loss_sensitive)
                tar_tr_loss.append(tr_loss_target)

                torch.save(model.state_dict(), f'./results/unet.pth')
            
            else:
                model.eval()
                if phase == 'valid':
                    loader = valid_loader
                else:
                    loader = test_loader
                    if epoch<epochs-1:
                        continue
                with torch.no_grad():
                    for data, target in tqdm(loader):
                        bilabel = target['bilabel'].float()
                        label = target['label'].float()
                        bbox = target['bbox']
                        mask = target['mask']
                        data, label, bilabel, mask = data.to(device), label.to(device), bilabel.to(device), mask.to(device)

                        attacked_data = model(data, 'bbox', mask)

                        output_target = classifier_target(attacked_data)
                        output_sensitive = classifier_sensitive(attacked_data)

                        loss_target = criterion(output_target, bilabel)
                        loss_sensitive = criterion(output_sensitive, label)
                        loss = loss_target+max(0,30-loss_sensitive)

                        running_loss_target += loss_target # sum up batch loss
                        running_loss_sensitive += loss_sensitive # sum up batch loss
                        running_ap_target += get_ap_score(torch.Tensor.cpu(bilabel).detach().numpy(), torch.Tensor.cpu(m(output_target)).detach().numpy()) 
                        running_ap_sensitive += get_ap_score(torch.Tensor.cpu(label).detach().numpy(), torch.Tensor.cpu(m(output_sensitive)).detach().numpy()) 


                        del data, target, output_target, output_sensitive, bilabel, label, bbox, mask
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    num_samples = float(len(loader.dataset))
                    tr_loss_target = running_loss_target.item()/num_samples
                    tr_loss_sensitive = running_loss_sensitive.item()/num_samples
                    tr_map_target = running_ap_target/num_samples
                    tr_map_sensitive = running_ap_sensitive/num_samples
                    print(f'loss_target{tr_loss_target}\tloss_sensitive{tr_loss_sensitive}')
                    print(f'acc_target{tr_map_target}\tacc_sensitive{tr_map_sensitive}')

                    sen_val_loss.append(tr_loss_sensitive)
                    tar_val_loss.append(tr_loss_target)

        plot_history([sen_tr_loss,tar_tr_loss,sen_val_loss,tar_val_loss],['sen_tr_loss','tar_tr_loss','sen_val_loss','tar_val_loss'],'loss','train_generator.png',c_list=['r--','g--','r','g'])


    model.train()
    
