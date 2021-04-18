import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import  transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets.attack_dataset import *  
def get_datasets(dataset_name="mnist"):
    """
    Loads clean original datasets.
    INPUT:
        dataset_name (str) : mnist or cifar10
    OUTPUT:
        train_data, test_data (datasets) : the two test/train portions of the original datasets
    """
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'mnist':
        train_data = datasets.MNIST(root='./data', train=True, transform=transform,download=True)
        test_data = datasets.MNIST(root='./data', train=False,transform=transform,download=True)

    elif dataset_name == 'cifar10':
        train_data =  datasets.CIFAR10(root='./data', train=True, transform=transform,download=True)
        test_data = datasets.CIFAR10(root='./data', train=False, transform=transform,download=True)
    return  train_data, test_data

def get_dataloaders(batch_size=32,dataset_name="mnist"):
    """
    Return the splits portions for shadow and target models 
    INPUT:
        batch_size (int) : batch size of the loader (default 32)
    OUTPUT:
    """
    split_portion = .25
    random_seed = 20

    train_data, test_data = get_datasets(dataset_name=dataset_name)
    
    # Creating data indices for training  splits:
    train_size = len(train_data)
    indices = list(range(train_size))
    split1 = int(np.floor(split_portion *  train_size))

    train_shadow1, outshadow1 = indices[:split1], indices[split1:2*split1]
    target_member1, target_N_member1 =  indices[2*split1:3*split1], indices[3*split1:]

    # Creating PT data samplers and loaders:
    train_shadow1_sampler = SubsetRandomSampler(train_shadow1)
    outshadow1_sampler = SubsetRandomSampler(outshadow1)
    target_member1_sampler = SubsetRandomSampler(target_member1)
    target_N_member1_sampler = SubsetRandomSampler(target_N_member1)

    shadow_train_loader = DataLoader(train_data, batch_size=batch_size, 
                                           sampler=train_shadow1_sampler)
    out_shadow_loader = DataLoader(train_data, batch_size=batch_size,
                                                sampler=outshadow1_sampler)

    target_train_member_loader = DataLoader(train_data, batch_size=batch_size, 
                                           sampler=target_member1_sampler)
    target_train_NMemember_loader = DataLoader(train_data, batch_size=batch_size,
                                                sampler=target_N_member1_sampler)

    # For test set
    
    test_loader = DataLoader(test_data, batch_size=batch_size)
   
    return {
        'DShadow_train' : shadow_train_loader,
        'DShadow_out' : out_shadow_loader,
        'target_train' : target_train_member_loader, # Used to train the target model
        'target_eval' : target_train_NMemember_loader, # Not part of the target model training
        'test_loader' : test_loader
    }


def get_attack_train_test_loaders(dataset_train_path,dataset_test_path,batch_size=32):
    
    train_dataset = Attack_dataset(dataset_train_path)
    test_dataset = Attack_dataset(dataset_test_path)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return {
        'train_loader' : train_loader,
        'test_loader' : test_loader
    }

