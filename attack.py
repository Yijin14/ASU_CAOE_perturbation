import torch
import numpy as np
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
from dataset import PascalVOC_Dataset
import torch.optim as optim
from train import train_classifier, test_classifier, train_generator
from utils import encode_labels, plot_history
from unet import MaskedUNet
import os
import sys
import torch.utils.model_zoo as model_zoo
import utils
from torchsummary import summary


def step0(train_loader,valid_loader,test_loader, model_name, num, lr, epochs, batch_size = 16, download_data = False, save_results=False, num_class=13, clean=True):
    model_dir = os.path.join("./models", model_name)
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir) 
    
    model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    }
    
    model_collections_dict = {
            "resnet18": models.resnet18(),
            "resnet34": models.resnet34(),
            "resnet50": models.resnet50()
            }
    
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Load pretrained model
    classifier = model_collections_dict[model_name]
    classifier.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    classifier.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    num_ftrs = classifier.fc.in_features
    classifier.fc = torch.nn.Linear(num_ftrs, num_class)
    print(f'Number of category is {classifier.fc.out_features}')
    classifier.to(device)

    optimizer = optim.SGD([   
            {'params': list(classifier.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(classifier.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
            ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    if clean:
        generator = None
    else:
        generator = MaskedUNet(c=3).to(device)
        network_state_dict = torch.load(f'./results/unet-1.pth')
        generator.load_state_dict(network_state_dict)
    
    # Imagnet values
    # mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    # std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    # loss, ap= test_classifier(num_class, classifier, device, test_loader, returnAllScores=False)
    #---------------Train your classifier here---------------------------------------
    loss, ap= test_classifier(num_class, classifier, generator, device, test_loader, returnAllScores=False)

    trn_hist, val_hist = train_classifier(num_class, classifier, generator, device, optimizer, scheduler, train_loader, valid_loader, model_dir, num, epochs, 'log_file')
    torch.cuda.empty_cache()

    plot_history([trn_hist[0], val_hist[0]], ['train','val'], "Loss", os.path.join(model_dir, "loss-{}".format(num)))
    plot_history([trn_hist[1], val_hist[1]], ['train','val'], "Accuracy", os.path.join(model_dir, "accuracy-{}".format(num)))    

    if clean:
        print('Save classifier...')
        torch.save(classifier.state_dict(), f'./results/model_{num_class}_clean.pth')
    
    #---------------Test your classifier here---------------------------------------
    if save_results:
        loss, ap, scores, gt = test_classifier(num_class, classifier, generator, device, test_loader, returnAllScores=True)
        
        gt_path, scores_path, scores_with_gt_path = os.path.join(model_dir, "gt-{}.csv".format(num)), os.path.join(model_dir, "scores-{}.csv".format(num)), os.path.join(model_dir, "scores_wth_gt-{}.csv".format(num))
        
        utils.save_results(test_loader.dataset.images, gt, utils.object_categories, gt_path)
        utils.save_results(test_loader.dataset.images, scores, utils.object_categories, scores_path)
        utils.append_gt(gt_path, scores_path, scores_with_gt_path)
        
        utils.get_classification_accuracy(gt_path, scores_path, os.path.join(model_dir, "clf_vs_threshold-{}.png".format(num)))
        
        # return loss, ap
    
    else:
        loss, ap= test_classifier(num_class, classifier, generator, device, test_loader, returnAllScores=False)
        
        # return loss, ap
    print(f'train loss {trn_hist[0]}')
    print(f'val loss {val_hist[0]}')
    print(f'train acc {trn_hist[1]}')
    print(f'val acc {val_hist[1]}')
    return trn_hist, val_hist
    

def step1(train_loader,valid_loader,test_loader, model_name, lr=0.01, epochs=3, batch_size = 16, download_data = False, save_results=False, target_class=2, sensitive_class=13, from_ckp=False):
    model_collections_dict = {
            "resnet18": models.resnet18(),
            "resnet34": models.resnet34(),
            "resnet50": models.resnet50()
            }

    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    classifier_target = models.resnet34()
    classifier_target.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    num_ftrs = classifier_target.fc.in_features
    classifier_target.fc = torch.nn.Linear(num_ftrs, target_class)
    classifier_sensitive = models.resnet34()
    classifier_sensitive.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    classifier_sensitive.fc = torch.nn.Linear(num_ftrs, sensitive_class)
    network_state_dict = torch.load(f'./results/model_{target_class}_clean.pth')
    classifier_target.load_state_dict(network_state_dict)
    network_state_dict = torch.load(f'./results/model_{sensitive_class}_clean.pth')
    classifier_sensitive.load_state_dict(network_state_dict)

    classifier_target.to(device)
    classifier_sensitive.to(device)

    generator = MaskedUNet(c=3).to(device)
    optimizer = optim.SGD(generator.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    if from_ckp==True:
        network_state_dict = torch.load(f'./results/unet-1.pth')
        generator.load_state_dict(network_state_dict)



    #---------------Train your generator here---------------------------------------
    train_generator(classifier_target, classifier_sensitive, generator, device, optimizer, scheduler, train_loader, valid_loader, test_loader, epochs, 'log_file')

    print('Save generator...')
    torch.save(generator.state_dict(), f'./results/unet.pth')





if __name__ == '__main__':
    data_dir = '../datasets/pascal/'
    download_data = False
    batch_size = 16

    transformations = transforms.Compose([utils.my_Padder(),transforms.ToTensor(),])
    transformations_valid = transforms.Compose([utils.my_Padder(),transforms.ToTensor(),])
    transformations_test = transforms.Compose([utils.my_Padder(),transforms.ToTensor(),])

    # Create dataloader
    dataset_train = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='train', 
                                      download=download_data, 
                                      transform=transformations, 
                                      target_transform=encode_labels)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    
    dataset_valid = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=download_data, 
                                      transform=transformations_valid, 
                                      target_transform=encode_labels)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=4)

    dataset_test = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=download_data, 
                                      transform=transformations_test, 
                                      target_transform=encode_labels)
    test_loader = DataLoader(dataset_test, batch_size=int(batch_size/4), num_workers=0, shuffle=False)

    def visualize():
        classifier2 = models.resnet34()
        num_ftrs = classifier2.fc.in_features
        classifier2.fc = torch.nn.Linear(num_ftrs, 2)
        classifier10 = models.resnet34()
        classifier10.fc = torch.nn.Linear(num_ftrs, 13)
        network_state_dict = torch.load(f'./results/model_{2}_clean.pth')
        classifier2.load_state_dict(network_state_dict)
        network_state_dict = torch.load(f'./results/model_{13}_clean.pth')
        classifier10.load_state_dict(network_state_dict)

        generator = MaskedUNet(c=3)#.to('cpu')
        network_state_dict = torch.load(f'./results/unet-11.pth')
        generator.load_state_dict(network_state_dict)

        transformations_test = transforms.Compose([#transforms.Resize(330), 
                                              #   transforms.FiveCrop(300), 
                                              utils.my_Padder((256,256,256)),
                                              transforms.ToTensor()
                                              #   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                              #   transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                                ])
        dataset_test = PascalVOC_Dataset('../datasets/pascal/',
                                            year='2012', 
                                            image_set='val', 
                                            download=False, 
                                            transform=transformations_test, 
                                            target_transform=encode_labels)
        test_loader = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)

        utils.visualize(classifier2, classifier10, generator, test_loader, num=3)
        # exit()

    # step0(train_loader,valid_loader,test_loader, "resnet34", num=13, lr = [1.5e-4, 5e-2], epochs = 6, batch_size=16, download_data=False, save_results=False, num_class=13)
    # step0(train_loader,valid_loader,test_loader, "resnet34", num=2, lr = [1.5e-4, 5e-2], epochs = 6, batch_size=16, download_data=False, save_results=False, num_class=2)

    # flag=True
    # while flag:
    #     print('training unet...')
    #     try:
    #         step1('../datasets/pascal/', "resnet34", epochs=5, batch_size=8, from_ckp=True)
    #         flag = False
    #     except:
    #       error = sys.exc_info()
    #       if error[0] == KeyboardInterrupt:
    #           print('EXIT')
    #           exit()
          

    step1(train_loader,valid_loader,test_loader, "resnet34", epochs=6, batch_size=8)#, from_ckp=True)

    visualize()

    # step0(train_loader,valid_loader,test_loader, "resnet34", num=13-2, lr = [1.5e-4, 5e-2], epochs = 6, batch_size=8, download_data=False, save_results=False, num_class=13, clean=False)
    # step0(train_loader,valid_loader,test_loader, "resnet34", num=2-2, lr = [1.5e-4, 5e-2], epochs = 6, batch_size=8, download_data=False, save_results=False, num_class=2, clean=False)
