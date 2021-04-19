import matplotlib.pyplot as plt
from torch import nn, optim
import torch
import torch.cuda as cuda_device
from pandas import DataFrame
import os
from sklearn.metrics import precision_score,recall_score , accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def display_batch(dataloader,Nbr_images = 6, dataset_name ="mnist"):
    """
    INPUT:
        dataloader (Dataloader) : the data loader of the images to display from
        Nbr_images (int) : number of images to display ( default 6 )
        dataset_name (str) : mnist or Cifar10
    """
    examples = enumerate(dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.cpu()
    
    fig = plt.figure()
    for i in range(Nbr_images):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        img = example_data[i].view(32,32,3) if dataset_name == "cifar10" else example_data[i][0]
        plt.imshow(img, cmap='gray', interpolation='none')
        #plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig

def train(train_dataloader, optimizer,model,n_epochs=50, model_path="./checkpoints/model.pth"):
    """
    Train a given model
    INPUT:
        train_dataloader ( Dataloader) : the loader of the training set
        optimizer  ( Optim) : the optimizer generally Adam
        model ( Pytorch nn.Module) : the model to be trained 
        n_epochs (int) : the number of times we loop through batches
        model_path  (str) : the path where to store the checkpoint 
    OUTPUT:
        loss_scores (list) : the list of loss for each epoch
    """
    loss_scores = []
    if cuda_device.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    model.train()
    for epo in range(n_epochs):
        for idx, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.float()
            x = x.cuda()
            y = y.cuda()
            output = model(x) 
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
         
        training_loss = running_loss/len(train_dataloader)
        print("Epoch {} - Training loss: {}".format(epo+1, training_loss ))
        loss_scores.append(training_loss)
        running_loss = 0
    torch.save(model.state_dict(), model_path)
    return loss_scores

def eval_model(model, data_loader,attack=True):
    """
    Evaluate the trained model on a test set
    INPUT:
        model (nn.Module) : the model to be eveluated
        dataloader (Dataloader) : the loader of the test set

    OUTPUT:
       accuracy (float) : accuracy on test set
    """
    model.eval()
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    model = model.cpu()
    predictions = []
    targets = []
    for i , (data, target) in enumerate(data_loader):
        data = data.float()
        #if torch.cuda.is_available():
            #data = data.cuda()
            #target = target.cuda()
        data = data.cpu()
        output = model(data)
        target = target.cpu()
        o = torch.argmax(output, dim=1)
        target = target.cpu()
        o = o.detach().tolist()
        predictions = predictions +o  
        t = target.detach().tolist()
        targets = targets +  t
       
        loss += criterion(output, target)
        #result = torch.eq(torch.argmax(output, dim=1), target)
        #correct += result.sum().item()
        #t = target
        #print(output)
    t = np.asarray(targets)
    p = np.asarray(predictions).round()     
    accuracy = accuracy_score(t,p)
    if attack == False:
        precision = precision_score(t,p,average='micro')
        recall = recall_score(t,p,average='micro')
    else:
        precision = precision_score(t,p,average='binary')
        recall = recall_score(t,p,average='binary')
    #accuracy = 100. * correct / len(data_loader.dataset)
    loss /= len(data_loader.dataset)
        
    print('\nAverage Val Loss: {:.4f}, Val Accuracy: ({:.3f}%) , precision:  {:.4f}, recall :  {:.4f}\n'.format(
        loss, accuracy,precision,recall ))
    return accuracy ,precision, recall 

def save_training_loss(loss,file_name):
    """
    Saves statistics about the training loss as an csv file
    INPUT:
        loss (list) : contains loss per epoch
        filename (str) : where to store the statistics
    """
    epochs = [x+1 for x in range(len(loss))] 
    liste = list(zip(epochs,loss))
    df = DataFrame(liste,columns=['epoch','training_loss'])
    outdir = "./results/"+ os.path.dirname(file_name) + "/" 
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, os.path.basename(file_name)) 
    df.to_csv(fullname, index=False)


def train_random_forests(x_train,y_train,n_estimators=1000):
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(x_train, y_train)
    return rf

def eval_random_forests(x_test,y_test,model):
    predictions = model.predict(x_test)
    errors = abs(predictions - y_test)
    accuracy = accuracy_score(y_test,predictions)

    precision = precision_score(y_test,predictions,average='binary')
    recall = recall_score(y_test,predictions,average='binary')
    print('\nAverage Val Loss: {:.4f}, Val Accuracy: ({:.3f}%) , precision:  {:.4f}, recall :  {:.4f}\n'.format(
        errors, accuracy,precision,recall ))
    return accuracy ,precision, recall 