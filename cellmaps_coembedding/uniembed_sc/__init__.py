import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from .architecture import *
import collections
import torch.nn.functional as F
import os
import csv


def write_embedding_dictionary_to_file(filepath, dictionary, dims):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        header_line = ['']
        header_line.extend([x for x in range(1, dims)])
        writer.writerow(header_line)
        for key, value in dictionary.items():
            row = [key]
            row.extend(value)
            writer.writerow(row)

def save_results(model, protein_dataset, data_wrapper, results_suffix = ''):
    
    resultsdir = data_wrapper.resultsdir + results_suffix
    model.eval()
    torch.save(model.state_dict(), '{}model.pth'.format(resultsdir))
    
    all_latents = dict()
    all_outputs = dict()
    for input_modality in  data_wrapper.modalities_dict.keys():
        all_latents[input_modality] = dict()
        for output_modality in data_wrapper.modalities_dict.keys():
            output_key = input_modality + '_' + output_modality
            all_outputs[output_key] = dict()

    with torch.no_grad():
        for i in np.arange(len(protein_dataset)):
            protein, mask, protein_name = protein_dataset[i]
            latents, outputs = model(protein) 
            for modality, latent in latents.items():
                if mask[modality] > 0:
                    all_latents[modality][protein_name] = latent.detach().cpu().numpy()
            for modality, output in outputs.items():
                input_modality = modality.split('_')[0]
                output_modality = modality.split('_')[1]
                if (mask[input_modality] > 0) & (mask[output_modality] > 0):
                    all_outputs[modality][protein_name] = output.detach().cpu().numpy()

    #save latent embeddings
    for modality, latents in all_latents.items():
        filepath = '{}_{}_latent.tsv'.format(resultsdir, modality)
        write_embedding_dictionary_to_file(filepath, latents, data_wrapper.latent_dim)
    
    #save reconstructed embeddings
    for modality, outputs in all_outputs.items():
        filepath = '{}_{}_reconstructed.tsv'.format(resultsdir, modality)
        output_modality = modality.split('_')[1]
        output_modality_dim = data_wrapper.modalities_dict[output_modality].input_dim
        write_embedding_dictionary_to_file(filepath, outputs, output_modality_dim)
                   
                    
def uniembed_fit_predict(resultsdir, modality_dataframes = [],
                     modality_names = [], 
                     test_subset = [], 
                     batch_size=14,
                     latent_dim=128,
                     n_epochs=500,
                     triplet_margin=1.0,
                     lambda_reconstruction = 1, 
                     lambda_triplet = 1,
                     lambda_l2 = 0.001,
                     l2_norm = False,
                     dropout=0, 
                     save_epoch = 50,
                     learn_rate = 1e-4,
                     hidden_size_1 = 512,
                     hidden_size_2 = 256,
                     save_update_epochs=False):
    
    
    source_file = open('{}.txt'.format(resultsdir), 'w')

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name()

    #if modality names doesn't match data size, create names with index
    num_data_modalities = len(modality_dataframes)
    if len(modality_names) != num_data_modalities:
        modality_names = ['modality_'.format(x) for x in np.arange(num_data_modalities)]

    data_wrapper = TrainingDataWrapper(modality_dataframes, modality_names, test_subset, device, l2_norm, dropout, 
                                       latent_dim, hidden_size_1, hidden_size_2, resultsdir)



    # create model, optimizer, trainloader 
    model = uniembed_nn(data_wrapper).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    protein_dataset = Protein_Dataset(data_wrapper.modalities_dict)
    train_loader = DataLoader(protein_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):

        # train
        total_loss = []
        total_reconstruction_loss = []
        total_triplet_loss = []
        total_l2_loss = []
        total_reconstruction_loss_by_modality = collections.defaultdict(list)#key: inputmodality_outputmodality
        total_triplet_loss_by_modality = collections.defaultdict(list) #key: modality

        model.train()
        # loop over all batches
        for step, (batch_proteins, batch_mask, _) in enumerate(train_loader):

            #pass through model
            latents, outputs = model(batch_proteins)
            
            batch_reconstruction_losses = torch.tensor([])
            batch_triplet_losses = torch.tensor([])
            batch_l2_losses = torch.tensor([])

            for input_modality in batch_proteins.keys():
                
                #get l2 loss
                for input_modality in batch_proteins.keys():
                    l2_loss = torch.norm(latents[input_modality], p=2, dim=1)
                    batch_l2_losses = torch.cat((batch_l2_losses, l2_loss))
                    
                    
                for output_modality in batch_proteins.keys():

                     #calculate pairwise sim
                    output_key = input_modality + '_' + output_modality
                    pairwise_sim = 1 - F.cosine_similarity(batch_proteins[input_modality].unsqueeze(1), outputs[output_key].unsqueeze(0), dim=2)                

                    #protein_present in both modalities mask
                    mask = (batch_mask[input_modality].bool()) & (batch_mask[output_modality].bool())
                    if torch.sum(mask) < 2: 
                        continue #not enough protein overlap across modalities

                    # get reconstruction_losses
                    reconstruction_loss = pairwise_sim[mask, mask].flatten()
                    batch_reconstruction_losses = torch.cat((batch_reconstruction_losses, reconstruction_loss))
                    total_reconstruction_loss_by_modality[input_modality + '_' + 
                                                          output_modality].append(torch.mean(reconstruction_loss).detach().cpu().numpy())
                    
                    #get triplet loss
                    if input_modality == output_modality:
                        continue #no triplet loss  if modalities match
                    positive_mask = torch.eye(pairwise_sim.shape[0]).bool().to(device)
                    negative_mask = torch.logical_not(positive_mask)
                    positive_dist = pairwise_sim[positive_mask]
                    #take closest negative for each row (set positives to infinity)
                    negative_dist = torch.min(torch.where(negative_mask, pairwise_sim, torch.inf), 1)[0]
                    #triplet is max of 0 or positive - negative
                    triplet_loss = torch.maximum(positive_dist - negative_dist + triplet_margin, torch.zeros(pairwise_sim.shape[0]))
                    batch_triplet_losses = torch.cat((batch_triplet_losses, triplet_loss))
                    total_triplet_loss_by_modality[input_modality + '_' + 
                                                   output_modality].append(torch.mean(triplet_loss).detach().cpu().numpy())
                    

            if (len(batch_reconstruction_losses) == 0 ) | (len(batch_triplet_losses) == 0):
                continue #didn't have any overlapping proteins in any modalities
                
            mean_reconstruction_loss = torch.mean(batch_reconstruction_losses)
            mean_triplet_loss = torch.mean(batch_triplet_losses)
            mean_l2_loss = torch.mean(batch_l2_losses)
            batch_total = lambda_reconstruction*mean_reconstruction_loss + lambda_triplet*mean_triplet_loss #+ lambda_l2*mean_l2_loss
            
            optimizer.zero_grad()
            batch_total.backward()
            optimizer.step()

            total_loss.append(batch_total.detach().cpu().numpy())
            total_reconstruction_loss.append(mean_reconstruction_loss.detach().cpu().numpy())
            total_triplet_loss.append(mean_triplet_loss.detach().cpu().numpy())
            total_l2_loss.append(mean_l2_loss.detach().cpu().numpy())

        #get result string wtith losses
        result_string = 'epoch:%d\ttotal_loss:%03.5f\treconstruction_loss:%03.5f\ttriplet_loss:%03.5f\tl2_loss:%03.5f\t' % (epoch, 
                                                                                                            np.mean(total_loss), 
                                                                                                    np.mean(total_reconstruction_loss), 
                                                                                                    np.mean(total_triplet_loss), np.mean(total_l2_loss))
        for modality, loss in total_reconstruction_loss_by_modality.items():
                result_string += '%s_reconstruction_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_triplet_loss_by_modality.items():
                result_string += '%s_triplet_loss:%03.5f\t' % (modality, np.mean(loss))
        print(result_string, file = source_file) 
        
        #save results at update epochs
        if (save_update_epochs) & (epoch%save_epoch == 0):
            save_results(model, protein_dataset, data_wrapper, results_suffix = '_epoch{}'.format(epoch))
           
    #save final results 
    save_results(model, protein_dataset, data_wrapper)
    source_file.close()

    return model