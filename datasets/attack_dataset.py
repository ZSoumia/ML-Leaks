import torch
from torch.utils.data import Dataset 
import pandas as pd 

class Attack_dataset(Dataset):
  def __init__(self, csv_file="attack_dataset.csv"):
    self.df = pd.read_csv(csv_file)


  def __len__(self): return len(self.df)

  def __getitem__(self, i): 
      row = self.df.iloc[i].tolist()
      return torch.tensor(row[:-1]),torch.tensor(row[-1]).long()# formatting is important I am supposing here that the last column corresponds to the label column