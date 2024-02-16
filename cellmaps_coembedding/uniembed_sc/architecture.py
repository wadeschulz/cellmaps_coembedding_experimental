### Classes used for coembedding 
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        return torch.from_numpy(sample).float()
    
    
class Modality():
    def __init__(self, training_dataframe, name, test_subset, transform, device):
        self.name = name
        self.all_features = transform(training_dataframe.values).to(device)
        self.all_labels = training_dataframe.index.values

        self.train_labels = list(set(self.all_labels) - set(test_subset))
        self.train_features = transform(training_dataframe.loc[self.train_labels].values).to(device)
        self.input_dim = self.train_features.shape[1]


class Protein_Dataset(Dataset):
    def __init__(self, modalities_dict):
        self.protein_dict = dict()
        self.mask_dict = dict()

        for modality in modalities_dict.values():
            for i in np.arange(len(modality.train_labels)):
                protein_name = modality.train_labels[i]
                protein_features = modality.train_features[i]
                # add to protein dictionary
                if protein_name not in self.protein_dict:
                    self.protein_dict[protein_name] = dict()
                    self.mask_dict[protein_name] = dict()
                #add modality to protein
                self.protein_dict[protein_name][modality.name] = protein_features
                self.mask_dict[protein_name][modality.name] = 1
       
        # add zeroes if not in dictionary
        for protein_name in self.protein_dict.keys():
            for modality in modalities_dict.values():
                if modality.name not in self.protein_dict[protein_name]:
                    self.protein_dict[protein_name][modality.name] = torch.zeros(modality.input_dim)
                    self.mask_dict[protein_name][modality.name] = 0
        
        self.protein_ids = dict(zip(np.arange(len(self.protein_dict.keys())), self.protein_dict.keys()))
        
    def __len__(self):
        return len(self.protein_dict)
    
    def __getitem__(self, index):
        item = self.protein_ids[index]
        return self.protein_dict[item], self.mask_dict[item], item

                
class TrainingDataWrapper():
    def __init__(self, modality_dataframes, modality_names, test_subset, device, l2_norm, dropout,latent_dim, hidden_size_1, hidden_size_2, 
                 resultsdir,):
        self.l2_norm = l2_norm
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.device = device 
        self.resultsdir = resultsdir
        self.transform=ToTensor()
        self.modalities_dict = dict()
        
        for i in np.arange(len(modality_names)):
            modality = Modality(modality_dataframes[i], modality_names[i], test_subset, self.transform, self.device)
            self.modalities_dict[modality_names[i]] = modality


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0) 
            
class uniembed_nn(nn.Module):
    def __init__(self, data_wrapper):
        super(uniembed_nn, self).__init__()
                
        self.l2_norm = data_wrapper.l2_norm
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
               
        # set up encoder and decoder for each modality
        for modality_name, modality in data_wrapper.modalities_dict.items():
            encoder = nn.Sequential(
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(modality.input_dim, data_wrapper.hidden_size_1),
                nn.ReLU(),
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(data_wrapper.hidden_size_1, data_wrapper.hidden_size_2),
                nn.Linear(data_wrapper.hidden_size_2, data_wrapper.latent_dim))
            encoder.apply(init_weights)   
            
            decoder = nn.Sequential(
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(data_wrapper.latent_dim, data_wrapper.hidden_size_2),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_2, data_wrapper.hidden_size_1),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_1, modality.input_dim))
            decoder.apply(init_weights)
            
            self.encoders[modality.name] = encoder
            self.decoders[modality.name] = decoder
            
    def forward(self, inputs):
        latents = dict()
        outputs = dict() 
        
        for modality_name, modality_values in inputs.items():
            
            latent = self.encoders[modality_name](modality_values)       
#             if self.l2_norm:
#                 latent = nn.functional.normalize(latent, p=2, dim=1)
            
            latents[modality_name] = latent
            
        for modality_name, modality_values in latents.items():
            for output_name, _ in inputs.items():
                outputs[modality_name + '_' + output_name] = self.decoders[output_name](modality_values)
        
        return inputs, outputs            
 