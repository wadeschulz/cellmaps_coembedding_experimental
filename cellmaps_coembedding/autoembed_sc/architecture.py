# Classes used for coembedding
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class ToTensor:
    """
    A class that converts a numpy ndarray to a torch Tensors.
    """
    def __call__(self, sample):
        """
        Convert the input numpy ndarray to a float Tensor.

        :param sample: The numpy array to be converted.
        :type sample: numpy.ndarray
        :return: Torch tensor of the input sample.
        :rtype: torch.Tensor
        """
        return torch.from_numpy(sample).float()


class Modality:
    """
    Represents a single modality of data, containing training features and labels.
    """
    def __init__(self, training_data, name, transform, device):
        """
        Initialize the Modality object with given training data, a name, a transformation, and the device.

        :param training_data: The data to use for training. Expects a list of lists where each sublist contains
                              the label followed by feature values.
        :type training_data: list
        :param name: The name of the modality.
        :type name: str
        :param transform: The transformation to apply to the data, converting it to a tensor.
        :type transform: callable
        :param device: The device to transfer the tensors to.
        :type device: torch.device
        """
        self.name = name

        embedding_data = []
        labels = []

        for xi in training_data:
            embedding_data.append(np.array([float(v) for v in xi[1:]]))
            labels.append(xi[0])

        self.train_labels = list(labels)
        self.train_features = transform(np.array(embedding_data)).to(device)
        self.input_dim = self.train_features.shape[1]


class Protein_Dataset(Dataset):
    """
    A dataset class for handling protein data across multiple modalities.
    """
    def __init__(self, wrapper):
        """
        Initialize the dataset using a dictionary of modalities.

        :param modalities_dict: A dictionary where keys are modality names and values are Modality objects.
        :type modalities_dict: dict
        """
        self.protein_dict = dict()
        self.mask_dict = dict()
        self.device = wrapper.device
        
        for modality in wrapper.modalities_dict.values():
            for i in np.arange(len(modality.train_labels)):
                protein_name = modality.train_labels[i]
                protein_features = modality.train_features[i]
                # add to protein dictionary
                if protein_name not in self.protein_dict:
                    self.protein_dict[protein_name] = dict()
                    self.mask_dict[protein_name] = dict()
                # add modality to protein
                self.protein_dict[protein_name][modality.name] = protein_features
                self.mask_dict[protein_name][modality.name] = 1

        # add zeroes if not in dictionary
        for protein_name in self.protein_dict.keys():
            for modality in wrapper.modalities_dict.values():
                if modality.name not in self.protein_dict[protein_name]:
                    self.protein_dict[protein_name][modality.name] = torch.zeros(modality.input_dim).to(wrapper.device)
                    self.mask_dict[protein_name][modality.name] = 0
                    
        self.protein_ids = dict(zip(np.arange(len(self.protein_dict.keys())), self.protein_dict.keys()))

    def __len__(self):
        """
        Return the total number of proteins in the dataset.

        :return: Total number of proteins.
        :rtype: int
        """
        return len(self.protein_dict)

    def __getitem__(self, index):
        """
        Retrieve the features and mask for a given protein by index.

        :param index: Index of the protein to retrieve.
        :type index: int
        :return: A tuple containing the protein's features, mask, and index.
        :rtype: tuple
        """
        item = self.protein_ids[index]
        return self.protein_dict[item], self.mask_dict[item], index


class TrainingDataWrapper:
    """
    Wraps training data for all modalities.
    """
    def __init__(self, modality_data, modality_names, device, l2_norm, dropout, latent_dim, hidden_size_1,
                 hidden_size_2,
                 resultsdir, ):
        """
        Initialize the wrapper with the given configuration.

        :param modality_data: List of training data for each modality.
        :type modality_data: list
        :param modality_names: Names for each modality.
        :type modality_names: list
        :param device: The device to use for tensor operations.
        :type device: torch.device
        :param l2_norm: Indicates if L2 normalization is used.
        :type l2_norm: bool
        :param dropout: The dropout rate for the neural network layers.
        :type dropout: float
        :param latent_dim: The dimensionality of the latent space.
        :type latent_dim: int
        :param hidden_size_1: The size of the first hidden layer.
        :type hidden_size_1: int
        :param hidden_size_2: The size of the second hidden layer.
        :type hidden_size_2: int
        :param resultsdir: The directory to save results.
        :type resultsdir: str
        """
        self.l2_norm = l2_norm
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.device = device
        self.resultsdir = resultsdir
        self.transform = ToTensor()
        self.modalities_dict = dict()

        for i in np.arange(len(modality_names)):
            modality = Modality(modality_data[i], modality_names[i], self.transform, self.device)
            self.modalities_dict[modality_names[i]] = modality


def init_weights(m):
    """
    Initialize weights for linear layers using Xavier normal distribution and biases to zero.

    :param m: The module to initialize.
    :type m: torch.nn.Module
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class uniembed_nn(nn.Module):
    """
    A neural network model for embedding proteins using multiple modalities.
    """
    def __init__(self, data_wrapper):
        """
        Initialize the model using a data wrapper that contains modality data configurations.

        :param data_wrapper: A wrapper containing configurations and data for all modalities.
        :type data_wrapper: TrainingDataWrapper
        """
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
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_2, data_wrapper.latent_dim))
            #    encoder.apply(init_weights)

            decoder = nn.Sequential(
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(data_wrapper.latent_dim, data_wrapper.hidden_size_2),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_2, data_wrapper.hidden_size_1),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_1, modality.input_dim))
            #     decoder.apply(init_weights)

            
            self.encoders[modality.name] = encoder
            self.decoders[modality.name] = decoder
            

    def forward(self, inputs):
        """
        Forward pass of the model, processing inputs through encoders and decoders.

        :param inputs: Dictionary of inputs where keys are modality names and values are corresponding tensors.
        :type inputs: dict
        :return: Tuple of dictionaries containing latent representations and outputs for all modalities.
        :rtype: tuple
        """
        latents = dict()
        outputs = dict()

        for modality_name, modality_values in inputs.items():

            latent = self.encoders[modality_name](modality_values)
            if self.l2_norm:
                if len(latent.shape) > 1:
                    latent = nn.functional.normalize(latent, p=2, dim=1)
                else:
                    latent = nn.functional.normalize(latent, p=2, dim=0)

            latents[modality_name] = latent

        for modality_name, modality_values in latents.items():
            for output_name, _ in inputs.items():
                outputs[modality_name + '_' + output_name] = self.decoders[output_name](modality_values)

        return latents, outputs

    def set_requires_grad(self, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for param in self.parameters():
            param.requires_grad = requires_grad


class Discriminator(nn.Module):
    def __init__(self,  data_wrapper, modality): 
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(modality.input_dim, data_wrapper.hidden_size_1),
            nn.ELU(),
            nn.Linear(data_wrapper.hidden_size_1, data_wrapper.hidden_size_2),
            nn.ELU(),
            nn.Linear(data_wrapper.hidden_size_2, 1))
    
    def forward(self, x):
        return self.discriminator(x)    
    
    def set_requires_grad(self, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for param in self.parameters():
            param.requires_grad = requires_grad

     
    
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
 
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None