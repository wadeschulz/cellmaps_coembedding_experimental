from .architecture import structured_embedding

from tqdm import tqdm
import phenograph
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .file_utils import *
from .df_utils import *
from .architecture import *
from .triplet_loss import *
import csv

# globals
source_file = ''
lambda_regul = 5
hard_loss = False
triplet_margin = 0.1
device = torch.device('cpu')
cur_muse_step = 0


def make_matrix_from_labels(labels):
    """
    Creates a symmetric matrix from cluster labels, where each entry (i, j) is set to 1 if elements i and j belong to
    the same cluster, otherwise 0.

    :param labels: Array of cluster labels.
    :type labels: array-like
    :return: A symmetric matrix indicating intra-cluster relationships.
    :rtype: numpy.ndarray
    """
    M = np.zeros((len(labels), len(labels)))
    for cluster in np.unique(labels):
        genes_in_cluster = np.where(labels == cluster)[0]
        for geneA in genes_in_cluster:
            for geneB in genes_in_cluster:
                M[geneA, geneB] = 1
    return M


def write_result_to_file(filepath, data, indexes):
    """
    Writes results to a tab-separated file with headers. Each row corresponds to the data array indexed with the
    corresponding index from 'indexes'.

    :param filepath: Path to the file where results will be saved.
    :type filepath: str
    :param data: Array of data to write.
    :type data: numpy.ndarray
    :param indexes: Index labels for each row of data.
    :type indexes: list
    """
    dims = data.shape[1]
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        header_line = ['']
        header_line.extend([x for x in range(1, dims)])
        writer.writerow(header_line)
        for i in np.arange(len(indexes)):
            row = [indexes[i]]
            row.extend(data[i])
            writer.writerow(row)


def train_model(model, optimizer, loader, label_x, label_y, epoch, lambda_super, train_name, train, device, log_fairops):
    """
    Trains a model using a DataLoader, tracking and computing various losses.

    :param model: Model to train.
    :type model: torch.nn.Module
    :param optimizer: Optimizer for updating model weights.
    :type optimizer: torch.optim.Optimizer
    :param loader: DataLoader for batch processing.
    :type loader: DataLoader
    :param label_x: Label matrix for input X.
    :type label_x: torch.Tensor
    :param label_y: Label matrix for input Y.
    :type label_y: torch.Tensor
    :param epoch: Current epoch number.
    :type epoch: int
    :param lambda_super: Weighting factor for triplet losses.
    :type lambda_super: float
    :param train_name: A name or tag for the training session, used in logging.
    :type train_name: str
    :param train: Boolean flag to determine if the model should be trained (True) or just evaluated (False).
    :type train: bool
    :param device: Device to run the training on (e.g., 'cuda' or 'cpu').
    :type device: torch.device
    """
    L_totals = []
    L_reconstruction_xs = []
    L_reconstruction_ys = []
    L_weights = []
    L_trip_batch_all_xs = []
    L_trip_batch_all_ys = []
    L_trip_batch_hard_xs = []
    L_trip_batch_hard_ys = []
    fraction_hard_xs = []
    fraction_hard_ys = []
    fraction_semi_xs = []
    fraction_semi_ys = []
    fraction_easy_xs = []
    fraction_easy_ys = []

    if log_fairops:
        global cur_muse_step
        import mlflow

    model.train()

    # loop over all batches
    for step, (batch_x_input, batch_y_input, batch_genes) in enumerate(loader):

        batch_label_x_input = label_x[batch_genes][:, batch_genes]
        batch_label_y_input = label_y[batch_genes][:, batch_genes]

        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(batch_x_input, batch_y_input)

        w_x = model.decoder_h_x.weight
        w_y = model.decoder_h_y.weight

        # calculate losses..

        # sparse penalty
        sparse_x = torch.sqrt(torch.sum(torch.sum(torch.square(w_x), axis=1)))
        sparse_y = torch.sqrt(torch.sum(torch.sum(torch.square(w_y), axis=1)))
        L_weight = sparse_x + sparse_y

        # triplet errors
        L_trip_batch_hard_x = batch_hard_triplet_loss(batch_label_x_input, latent, triplet_margin, device)
        L_trip_batch_hard_y = batch_hard_triplet_loss(batch_label_y_input, latent, triplet_margin, device)
        L_trip_batch_all_x, _ = batch_all_triplet_loss(batch_label_x_input, latent, triplet_margin, device)
        L_trip_batch_all_y, _ = batch_all_triplet_loss(batch_label_y_input, latent, triplet_margin, device)

        fraction_easy_x, fraction_semi_x, fraction_hard_x = fraction_triplets(batch_label_x_input, latent,
                                                                              triplet_margin, device)
        fraction_easy_y, fraction_semi_y, fraction_hard_y = fraction_triplets(batch_label_y_input, latent,
                                                                              triplet_margin, device)

        # reconstruction error
        L_reconstruction_x = torch.mean(torch.norm(reconstruct_x - batch_x_input))
        L_reconstruction_y = torch.mean(torch.norm(reconstruct_y - batch_y_input))

        L_total = lambda_super * (
                L_trip_batch_all_x + L_trip_batch_all_y) + lambda_regul * L_weight + L_reconstruction_x + L_reconstruction_y

        if hard_loss:
            L_total = lambda_super * (
                    L_trip_batch_hard_x + L_trip_batch_hard_y) + lambda_regul * L_weight + L_reconstruction_x + L_reconstruction_y

        if train == True:
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

        L_totals.append(L_total.detach().cpu().numpy())
        L_reconstruction_xs.append(L_reconstruction_x.detach().cpu().numpy())
        L_reconstruction_ys.append(L_reconstruction_y.detach().cpu().numpy())
        L_weights.append(L_weight.detach().cpu().numpy())
        L_trip_batch_hard_xs.append(L_trip_batch_hard_x.detach().cpu().numpy())
        L_trip_batch_hard_ys.append(L_trip_batch_hard_y.detach().cpu().numpy())
        L_trip_batch_all_xs.append(L_trip_batch_all_x.detach().cpu().numpy())
        L_trip_batch_all_ys.append(L_trip_batch_all_y.detach().cpu().numpy())
        fraction_hard_xs.append(fraction_hard_x.detach().cpu().numpy())
        fraction_hard_ys.append(fraction_hard_y.detach().cpu().numpy())
        fraction_semi_xs.append(fraction_semi_x.detach().cpu().numpy())
        fraction_semi_ys.append(fraction_semi_y.detach().cpu().numpy())
        fraction_easy_xs.append(fraction_easy_x.detach().cpu().numpy())
        fraction_easy_ys.append(fraction_easy_y.detach().cpu().numpy())

    print(
        train_name + "_epoch:%d\ttotal_loss:%03.5f\treconstruction_loss_x:%03.5f\treconstruction_loss_y:%03.5f\tsparse_penalty:%03.5f\tx_triplet_loss_batch_hard:%03.5f\ty_triplet_loss_batch_hard:%03.5f\tx_triplet_loss_batch_all:%03.5f\ty_triplet_loss_batch_all:%03.5f\tx_fraction_hard:%03.5f\ty_fraction_hard:%03.5f\tx_fraction_semi:%03.5f\ty_fraction_semi:%03.5f\tx_fraction_easy:%03.5f\ty_fraction_easy:%03.5f"
        % (epoch, np.mean(L_totals), np.mean(L_reconstruction_xs), np.mean(L_reconstruction_ys), np.mean(L_weights),
           np.mean(L_trip_batch_hard_xs), np.mean(L_trip_batch_hard_ys), np.mean(L_trip_batch_all_xs),
           np.mean(L_trip_batch_all_ys), np.mean(fraction_hard_xs), np.mean(fraction_hard_ys),
           np.mean(fraction_semi_xs), np.mean(fraction_semi_ys), np.mean(fraction_easy_xs), np.mean(fraction_easy_ys)),
        file=source_file)
    
    if log_fairops:
        mlflow.log_metrics({
            f"{train_name}_epoch": epoch,
            "total_loss": np.mean(L_totals),
            "reconstruction_loss": np.mean(L_reconstruction_xs),
            "reconstruction_loss_y": np.mean(L_reconstruction_ys),
            "sparse_penalty": np.mean(L_weights),
            "x_triplet_loss_batch_hard": np.mean(L_trip_batch_hard_xs),
            "y_triplet_loss_batch_hard": np.mean(L_trip_batch_hard_ys),
            "x_triplet_loss_batch_all": np.mean(L_trip_batch_all_xs),
            "y_triplet_loss_batch_all": np.mean(L_trip_batch_all_ys),
            "x_fraction_hard": np.mean(fraction_hard_xs),
            "y_fraction_hard": np.mean(fraction_hard_ys),
            "x_fraction_semi": np.mean(fraction_semi_xs),
            "y_fraction_semi": np.mean(fraction_semi_ys),
            "x_fraction_easy": np.mean(fraction_easy_xs),
            "y_fraction_easy": np.mean(fraction_easy_ys)
        }, step=cur_muse_step)
    cur_muse_step += 1

def muse_fit_predict(resultsdir,
                     modality_data=[],
                     modality_names=[],
                     name_index=[],
                     label_x=[],
                     label_y=[],
                     test_subset=[],
                     batch_size=64,
                     latent_dim=128,
                     n_epochs=500,
                     n_epochs_init=200,
                     lambda_regul=5,
                     lambda_super=5,
                     triplet_margin=0.1,
                     hard_loss=False,
                     l2_norm=True,
                     k=10,
                     dropout=0.25,
                     save_update_epochs=False,
                     log_fairops=False):
    """
    Fits a model using provided datasets and predicts outputs.

    :param resultsdir: Directory where results and model states are saved.
    :type resultsdir: str
    :param modality_data: List of datasets for different modalities (X and Y).
    :type modality_data: list of numpy.ndarray
    :param modality_names: Names of modalities.
    :type modality_names: list of str
    :param name_index: Index or names associated with the data samples.
    :type name_index: list
    :param label_x: Cluster labels or matrices for modality X.
    :type label_x: list
    :param label_y: Cluster labels or matrices for modality Y.
    :type label_y: list
    :param test_subset: Indices of the test subset.
    :type test_subset: list
    :param batch_size: Size of each data batch.
    :type batch_size: int
    :param latent_dim: Dimension of the latent space.
    :type latent_dim: int
    :param n_epochs: Total number of epochs for training.
    :type n_epochs: int
    :param n_epochs_init: Number of initial epochs for training without label updates.
    :type n_epochs_init: int
    :param lambda_regul: Regularization factor for the loss function.
    :type lambda_regul: float
    :param lambda_super: Supervision strength in loss function.
    :type lambda_super: float
    :param triplet_margin: Margin for triplet loss calculation.
    :type triplet_margin: float
    :param hard_loss: Flag to use hard triplet loss.
    :type hard_loss: bool
    :param l2_norm: Flag to use L2 normalization.
    :type l2_norm: bool
    :param k: Number of neighbors for clustering.
    :type k: int
    :param dropout: Dropout rate.
    :type dropout: float
    :param save_update_epochs: Flag to save model state at specified epoch intervals.
    :type save_update_epochs: bool
    :return: Model and embeddings as final outputs.
    :rtype: tuple
    """
    if log_fairops:
        import mlflow

    # get data
    data_x = modality_data[0]
    data_y = modality_data[1]
    num_data_modalities = len(modality_data)
    if len(modality_names) != num_data_modalities:
        modality_names = ['modality_'.format(x) for x in np.arange(num_data_modalities)]
    name_x = modality_names[0]
    name_y = modality_names[1]

    # parameter setting for neural network
    n_hidden = 128  # number of hidden node in neural network
    learn_rate = 1e-4  # learning rate in the optimization
    batch_size = 64  # number of cells in the training batch
    cluster_update_epoch = 50
    source_file = open('{}.txt'.format(resultsdir), 'w')

    if log_fairops:
        mlflow.log_params({
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "n_epochs": n_epochs,
            "n_epochs_init": n_epochs_init,
            "lambda_regul": lambda_regul,
            "lambda_super": lambda_super,
            "triplet_margin": triplet_margin,
            "hard_loss": hard_loss,
            "l2_norm": l2_norm,
            "k": k,
            "dropout": dropout,
            "n_hidden": n_hidden,
            "learn_rate": learn_rate,
            "batch_size": batch_size,
            "cluster_update_epoch": cluster_update_epoch,
            "save_update_epochs": save_update_epochs
        })

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name()

    # set globals  (same across all training)
    globals()['source_file'] = source_file
    globals()['lambda_regul'] = lambda_regul
    globals()['triplet_margin'] = triplet_margin
    globals()['hard_loss'] = hard_loss
    globals()['device'] = device
    global cur_muse_step

    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]
    feature_dim_y = data_y.shape[1]
    n_sample = data_x.shape[0]

    # transform inputs to tensor
    transform = ToTensor()
    data_x = transform(data_x).to(device)
    data_y = transform(data_y).to(device)

    # index names if none input
    if len(name_index) == 0:
        name_index = np.arange(n_sample)

    # remove test subset...
    train_subset = np.arange(n_sample)
    train_subset = list(set(train_subset) - set(test_subset))
    train_data_x = data_x[train_subset]
    train_data_y = data_y[train_subset]
    if len(label_x) > 0:
        label_x = label_x[train_subset]
    if len(label_y) > 0:
        label_y = label_y[train_subset]

    # create initial cluster labels if non input - only on training data
    create_label_x = False
    if len(label_x) == 0:
        label_x, _, mod_metric = phenograph.cluster(train_data_x.detach().cpu().numpy(), k=k, primary_metric='cosine')
        if log_fairops:
            mlflow.log_metric("x_mod_metric", mod_metric, step=cur_muse_step)
        label_x = transform(make_matrix_from_labels(label_x)).to(device)
        create_label_x = True
    else:
        if (len(label_x.shape) == 1) or (label_x.shape[1] == 1):
            label_x = transform(make_matrix_from_labels(label_x)).to(device)
        else:
            label_x = transform(label_x).to(device)

    create_label_y = False
    if len(label_y) == 0:
        label_y, _, mod_metric = phenograph.cluster(train_data_y.detach().cpu().numpy(), k=k, primary_metric='cosine')
        if log_fairops:
            mlflow.log_metric("y_mod_metric", mod_metric, step=cur_muse_step)
        label_y = transform(make_matrix_from_labels(label_y)).to(device)
        create_label_y = True
    else:
        if (len(label_y.shape) == 1) or (label_y.shape[1] == 1):
            label_y = transform(make_matrix_from_labels(label_y)).to(device)
        else:
            label_y = transform(label_y).to(device)

    # create model, optimizer, trainloader
    model = structured_embedding(feature_dim_x, feature_dim_y, latent_dim, n_hidden, dropout, l2_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_loader = DataLoader(Protein_Dataset(train_data_x, train_data_y), batch_size=batch_size, shuffle=True)

    # INIT WITH JUST RECONSTRUCTION
    for epoch in range(n_epochs_init):
        model.train()
        train_model(model, optimizer, train_loader, label_x, label_y, epoch, 0, 'init_recon', True, device, log_fairops)

    #  INIT WITH TRIPLET LOSS AND RECONSTRUCTION, ORIGINAL LABELS
    for epoch in range(n_epochs_init):
        model.train()
        train_model(model, optimizer, train_loader, label_x, label_y, epoch, lambda_super, 'init_both', True, device, log_fairops)

    latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(train_data_x, train_data_y)

    update_label_x = label_x
    update_label_y = label_y
    if create_label_x:
        update_label_x, _, mod_metric = phenograph.cluster(latent_x.detach().cpu().numpy(), k=k, primary_metric='cosine')
        if log_fairops:
            mlflow.log_metric("x_mod_metric", mod_metric, step=cur_muse_step)
        update_label_x = transform(make_matrix_from_labels(update_label_x)).to(device)
    if create_label_y:
        update_label_y, _, mod_metric = phenograph.cluster(latent_y.detach().cpu().numpy(), k=k, primary_metric='cosine')
        if log_fairops:
            mlflow.log_metric("y_mod_metric", mod_metric, step=cur_muse_step)
        update_label_y = transform(make_matrix_from_labels(update_label_y)).to(device)

    # TRAIN WITH LABELS
    for epoch in range(n_epochs):
        model.train()
        train_model(model, optimizer, train_loader, update_label_x, update_label_y, epoch, lambda_super, 'train', True,
                    device, log_fairops)

        if epoch % cluster_update_epoch == 0:
            model.eval()
            with torch.no_grad():
                latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)

            if save_update_epochs:
                torch.save(model.state_dict(), '{}_{}.pth'.format(resultsdir, epoch))
                write_result_to_file('{}_latent_{}.txt'.format(resultsdir, epoch), latent.detach().cpu().numpy(),
                                     name_index)
                write_result_to_file('{}_reconstruct_{}_{}.txt'.format(resultsdir, name_x, epoch),
                                     reconstruct_x.detach().cpu().numpy(), name_index)
                write_result_to_file('{}_reconstruct_{}_{}.txt'.format(resultsdir, name_y, epoch),
                                     reconstruct_y.detach().cpu().numpy(), name_index)
                write_result_to_file('{}_latent_{}_{}.txt'.format(resultsdir, name_x, epoch),
                                     latent_x.detach().cpu().numpy(), name_index)
                write_result_to_file('{}_latent_{}_{}.txt'.format(resultsdir, name_y, epoch),
                                     latent_y.detach().cpu().numpy(), name_index)

            # update clusters (only on training data)
            if create_label_x:
                train_latent_x = latent_x[train_subset]
                update_label_x, _, mod_metric = phenograph.cluster(train_latent_x.detach().cpu().numpy(), k=k,
                                                          primary_metric='cosine')
                if log_fairops:
                    mlflow.log_metric("x_mod_metric", mod_metric, step=cur_muse_step)
                update_label_x = transform(make_matrix_from_labels(update_label_x)).to(device)
            if create_label_y:
                train_latent_y = latent_y[train_subset]
                update_label_y, _, mod_metric = phenograph.cluster(train_latent_y.detach().cpu().numpy(), k=k,
                                                          primary_metric='cosine')
                if log_fairops:
                    mlflow.log_metric("y_mod_metric", mod_metric, step=cur_muse_step)
                update_label_y = transform(make_matrix_from_labels(update_label_y)).to(device)

    # SAVE FINAL RESULTS
    model.eval()
    with torch.no_grad():
        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)

    detached_embeddings = latent.detach().cpu().numpy()
    torch.save(model.state_dict(), '{}.pth'.format(resultsdir))
    write_result_to_file('{}_latent.txt'.format(resultsdir), latent.detach().cpu().numpy(), name_index)
    write_result_to_file('{}_reconstruct_{}.txt'.format(resultsdir, name_x),
                         reconstruct_x.detach().cpu().numpy(), name_index)
    write_result_to_file('{}_reconstruct_{}.txt'.format(resultsdir, name_y),
                         reconstruct_y.detach().cpu().numpy(), name_index)
    write_result_to_file('{}_latent_{}.txt'.format(resultsdir, name_x),
                         latent_x.detach().cpu().numpy(), name_index)
    write_result_to_file('{}_latent_{}.txt'.format(resultsdir, name_y),
                         latent_y.detach().cpu().numpy(), name_index)

    source_file.close()

    return model, detached_embeddings
