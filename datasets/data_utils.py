import torch.nn.functional as F
import torch
import pandas as pd 

def get_shadow_feature_vector(shadow_model,batch,member=False):
    """
    Get the top 3 probabilities of batch data passed to a shadow model and label them 
    (labelling follows the following strategy : 
    if batch is part of the training set of the target aka DtargetTrain then it's 1 otherwise it's 0)
    INPUT:
        shadow_model (nn.Module) : the shadow model trained
        batch (Tensor) : a batch size of unlabelled mnist
        member (boolean) : whether the data in the tensor were used to train the tagret or not
    OUTPUT:
        top3_probs,labels (list(tensor,tensor)) : top3 probabilities of each entry in the batch labbeled 
    """
    batch = batch.cpu()
    shadow_model = shadow_model.cpu()
    output = shadow_model(batch)
    probs = F.softmax(output,dim=-1)
    sorted_probs, indices  = torch.sort(probs,descending=True)
    top3_probs = torch.narrow(sorted_probs,dim=1,start=0,length=3)
    if member: 
        labels = torch.ones(top3_probs.shape[0])
    else:
        labels = torch.zeros(top3_probs.shape[0])
    
    labels = torch.tensor(labels)

    return [top3_probs,labels]

def get_attack_trainset(shadow_model,shadow_train_loader, shadow_out_loader,file_path='data/attack_dataset.csv'):

    result = []
    df = pd.DataFrame()

    for idx, (x, y) in enumerate(shadow_train_loader):
        feature_vect = get_shadow_feature_vector(shadow_model,x,member=True)
        result.append(feature_vect)
        df2 = pd.DataFrame({"top1": feature_vect[0][:,0].tolist(),
                            "top2":feature_vect[0][:,1].tolist(),
                            "top3":feature_vect[0][:,2].tolist(),
                            "label": feature_vect[1].tolist(),
                            })
        
        df = df.append(df2)
  
    for idx, (x, y) in enumerate(shadow_out_loader):
        feature_vect = get_shadow_feature_vector(shadow_model,x,member=False)
        result.append(feature_vect)
        df2 = pd.DataFrame({"top1": feature_vect[0][:,0].tolist(),
                            "top2":feature_vect[0][:,1].tolist(),
                            "top3":feature_vect[0][:,2].tolist(),
                            "label": feature_vect[1].tolist(),
                            })
        df = df.append(df2)
    df.to_csv(file_path,index=False)
    return df