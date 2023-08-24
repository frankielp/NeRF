import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from func import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        Initialize the NeRF model with the given parameters.

        Args:
            D (int): Number of layers in the NeRF model.
            W (int): Width of the hidden layers.
            input_ch (int): Number of input channels (for the scene input).
            input_ch_views (int): Number of input channels (for the view direction input).
            output_ch (int): Number of output channels.
            skips (list): List of skip connections indices.
            use_viewdirs (bool): Flag to use view direction information.
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Define the point (scene input) linear layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # Define the view direction linear layers
        # According to the official code release
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # Define the feature, alpha, and RGB linear layers if using view direction information
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            # Define the output linear layer
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        Forward pass through the NeRF model.

        Args:
            x (tensor): Input tensor containing both scene input and view direction information.
                        Shape should be (batch_size, input_ch + input_ch_views).

        Returns:
            outputs (tensor): Output tensor containing the predicted RGB color and alpha values.
                              Shape will be (batch_size, output_ch).
        """
        # Split the input tensor into scene input and view direction information
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        # Iterate through the point (scene input) linear layers
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # Calculate alpha and feature values from the point linear layers
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            # Iterate through the view direction linear layers
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            # Calculate RGB values from the view direction linear layers
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            # Calculate output values using the output linear layer
            outputs = self.output_linear(h)

        return outputs

# Positional encoding (section 5.1)
class Embedder:
    """
    Class to create an embedding function that transforms input data into a higher-dimensional representation.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        Create the embedding functions based on the provided configuration parameters.
        The embedding functions are defined based on the input dimensions, frequency bands, and periodic functions.
        """
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        # Include the original input if specified
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        # Calculate frequency bands based on log sampling or linear sampling
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        # Create embedding functions for each frequency band and periodic function
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    """
    Get an embedding function based on the given configuration.

    Args:
        multires (int): log2 of max freq for positional encoding (3D location)
        i (int): set 0 for default positional encoding, -1 for none

    Returns:
        embed (function): The embedding function.
        out_dim (int): The dimension of the output embedding.
    """
    if i == -1:
        # Return an identity function if i is -1
        return nn.Identity(), 3
    
    # Define configuration parameters for the embedding
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    # Create an Embedder object with the specified configuration
    embedder_obj = Embedder(**embed_kwargs)
    
    # Define the embedding function using the created Embedder object
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    
    # Return the embedding function and the output dimension of the embedding
    return embed, embedder_obj.out_dim



def create_nerf(args):
    """
    Instantiate the NeRF model along with associated configurations.

    Args:
        args (Namespace): Parsed arguments containing configuration settings.

    Returns:
        render_kwargs_train (dict): Configuration settings for training.
        render_kwargs_test (dict): Configuration settings for testing.
        start (int): The starting iteration step.
        grad_vars (list): List of model parameters for optimization.
        optimizer (torch.optim): Optimizer for model training.
    """
    # Get the embedding function and input channel count
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # Determine the output channel based on the N_importance setting - number of additional fine samples per ray
    output_ch = 5 if args.N_importance > 0 else 4

    # Skips connection for the NeRF model
    skips = [4]

    # Instantiate the NeRF model
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    # Collect model parameters for optimization
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # Define the network query function
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                         embed_fn=embed_fn,
                                                                         embeddirs_fn=embeddirs_fn,
                                                                         netchunk=args.netchunk)

    # Create the optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer



if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    model = NeRF()

  