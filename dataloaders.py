import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import  transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

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

    shadow_train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           sampler=train_shadow1_sampler)
    out_shadow_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                sampler=outshadow1_sampler)

    target_train_member_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           sampler=target_member1_sampler)
    target_train_NMemember_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                sampler=target_N_member1_sampler)

    # For test set
    split_portion = .5
    # Creating data indices for training  splits:
    test_size = len(test_data)
    indices = list(range(train_size))
    split1 = int(np.floor(split_portion *  test_size))

    test_shadow2, target_member2 = indices[:split1], indices[split_portion:]
    # Creating PT data samplers and loaders:
    test_shadow2_sampler = SubsetRandomSampler(test_shadow2)
    
    target_member2_sampler = SubsetRandomSampler(target_member2)

    shadow_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                           sampler=test_shadow2_sampler)
    target_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                           sampler=target_member2_sampler)
   

    return {
        'DShadow_train' : shadow_train_loader,
        'Dshadow_train_test'
    }
