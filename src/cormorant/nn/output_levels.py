import torch
import torch.nn as nn

from cormorant.nn import BasicMLP
from cormorant.so3_lib import cat

############# Get Scalars #############

class GetScalarsAtom(nn.Module):
    """
    Construct a set of scalar feature vectors for each atom by using the
    covariant atom :class:`SO3Vec` representations at various levels.

    Parameters
    ----------
    tau_levels : :class:`list` of :class:`SO3Tau`
        Multiplicities of the output :class:`SO3Vec` at each level.
    full_scalars : :class:`bool`, optional
        Construct a more complete set of scalar invariants from the full
        :class:`SO3Vec` (``true``), or just use the :math:``\ell=0`` component
        (``false``).
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, tau_levels, full_scalars=True, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.maxl = max([len(tau) for tau in tau_levels]) - 1

        signs_tr = [torch.pow(-1, torch.arange(-m, m+1.)) for m in range(self.maxl+1)]
        signs_tr = [torch.stack([s, -s], dim=-1) for s in signs_tr]
        self.signs_tr = [s.view(1, 1, 1, -1, 2).to(device=device, dtype=dtype) for s in signs_tr]

        split_l0 = [tau[0] for tau in tau_levels]
        split_full = [sum(tau) for tau in tau_levels]

        self.full_scalars = full_scalars
        if full_scalars:
            self.num_scalars = sum(split_l0) + sum(split_full)
            self.split = split_l0 + split_full
        else:
            self.num_scalars = sum(split_l0)
            self.split = split_l0

        print('Number of scalars at top:', self.num_scalars)

    def forward(self, reps_all_levels):
        """
        Forward step for :class:`GetScalarsAtom`

        Parameters
        ----------
        reps_all_levels : :class:`list` of :class:`SO3Vec`
            List of covariant atom features at each level

        Returns
        -------
        scalars : :class:`torch.Tensor`
            Invariant scalar atom features constructed from ``reps_all_levels``
        """

        reps = cat(reps_all_levels)

        scalars = reps[0]

        if self.full_scalars:
            scalars_tr  = [(sign*part*part.flip(-2)).sum(dim=(-1, -2), keepdim=True) for part, sign in zip(reps, self.signs_tr)]
            scalars_mag = [(part*part).sum(dim=(-1, -2), keepdim=True) for part in reps]

            scalars_full = [torch.cat([s_tr, s_mag], dim=-1) for s_tr, s_mag in zip(scalars_tr, scalars_mag)]

            scalars = [scalars] + scalars_full

            scalars = torch.cat(scalars, dim=-3)

        return scalars


class GetScalarsEdge(nn.Module):
    """
    Construct a set of scalar feature vectors for each edge.

    Parameters
    ----------
    tau_levels : :class:`list` of :class:`SO3Tau`
        Multiplicities of the output :class:`SO3Scalar` at each level.
    full_scalars : :class:`bool`, optional
        Construct a more complete set of scalar invariants from the full
        :class:`SO3Vec` (``true``), or just use the :math:``\ell=0`` component
        (``false``).
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, tau_levels, full_scalars=True, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.num_scalars = 2 * sum([sum(tau) for tau in tau_levels])
        # self.maxl = max([len(tau) for tau in tau_levels]) - 1

        # signs_tr = [torch.pow(-1, torch.arange(-m, m+1.)) for m in range(self.maxl+1)]
        # signs_tr = [torch.stack([s, -s], dim=-1) for s in signs_tr]
        # self.signs_tr = [s.view(1, 1, 1, -1, 2).to(device=device, dtype=dtype) for s in signs_tr]

        # split_l0 = [tau[0] for tau in tau_levels]
        # split_full = [sum(tau) for tau in tau_levels]

        # self.full_scalars = full_scalars
        # if full_scalars:
        #     self.num_scalars = sum(split_l0) + sum(split_full)
        #     self.split = split_l0 + split_full
        # else:
        #     self.num_scalars = sum(split_l0)
        #     self.split = split_l0

        print('Number of scalars at top:', self.num_scalars)

    def forward(self, reps_all_levels):
        """
        Forward step for :class:`GetScalarsAtom`

        Parameters
        ----------
        reps_all_levels : :class:`list` of :class:`SO3Vec`
            List of covariant atom features at each level

        Returns
        -------
        scalars : :class:`torch.Tensor`
            Invariant scalar atom features constructed from ``reps_all_levels``
        """

        reps = cat(reps_all_levels)
        individual_reps = [torch.cat((ri[..., 0], ri[..., 1]), dim=-1) for ri in reps]
        all_reps = torch.cat([ri for ri in individual_reps], dim=-1)
        return all_reps

        # scalars = reps[0]

        # if self.full_scalars:
        #     scalars_tr  = [(sign*part*part.flip(-2)).sum(dim=(-1, -2), keepdim=True) for part, sign in zip(reps, self.signs_tr)]
        #     scalars_mag = [(part*part).sum(dim=(-1, -2), keepdim=True) for part in reps]

        #     scalars_full = [torch.cat([s_tr, s_mag], dim=-1) for s_tr, s_mag in zip(scalars_tr, scalars_mag)]

        #     scalars = [scalars] + scalars_full

        #     scalars = torch.cat(scalars, dim=-3)


############# Output of network #############

class OutputLinear(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors. This is performed in a permutation invariant way
    by using a (batch-masked) sum over all atoms, and then applying a
    linear mixing layer to predict a single output.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    num_out: :class:`int`
        Number outputs to the network. 
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, num_out=1, bias=True, device=torch.device('cpu'), dtype=torch.float):
        super(OutputLinear, self).__init__()

        self.num_scalars = num_scalars
        self.bias = bias

        self.lin = nn.Linear(num_scalars, num_out, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputLinear`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        s = atom_scalars.shape
        atom_scalars = atom_scalars.view((s[0], s[1], -1)).sum(1)  # No masking needed b/c summing over atoms

        predict = self.lin(atom_scalars)

        predict = predict.squeeze(-1)

        return predict


class OutputPMLP(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors.

    This is peformed in a three-step process::

    (1) A MLP is applied to each set of scalar atom-features.
    (2) The environments are summed up.
    (3) Another MLP is applied to the output to predict a single learning target.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, num_mixed=64, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars
        self.num_mixed = num_mixed

        self.mlp1 = BasicMLP(2*num_scalars, num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_mixed, 1, num_hidden=1, activation=activation, device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputPMLP`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        # Reshape scalars appropriately
        atom_scalars = atom_scalars.view(atom_scalars.shape[:2] + (2*self.num_scalars,))

        # First MLP applied to each atom
        x = self.mlp1(atom_scalars)

        # Reshape to sum over each atom in molecules, setting non-existent atoms to zero.
        atom_mask = atom_mask.unsqueeze(-1)
        x = torch.where(atom_mask, x, self.zero).sum(1)

        # Prediction on permutation invariant representation of molecules
        predict = self.mlp2(x)

        predict = predict.squeeze(-1)

        return predict


class OutputEdgeLinear(nn.Module):
    """
    Runs a single Linear on the values on each edge, without communication.

    Parameters
    ----------
    num_channels_in : int
        Number of input channels for every edge
    num_out : int
        Number of output channels for every edge
    num_hidden : int
        Number of hidden layers to use
    layer_width : int
        width of the hidden layers.
    activation : str
        Activation function to use for the nonlinearity.
    device : pytorch device object
        Computer architecture the code is being run on.
    dtype : pytorch datatype
        Type of pytorch datatype
    """
    def __init__(self, num_channels_in, num_out=1, bias=True, device=torch.device('cpu'), dtype=torch.float):
        super(OutputEdgeLinear, self).__init__()

        print('num out in init', num_out)

        self.num_channels_in = num_channels_in
        self.lin = nn.Linear(num_channels_in, num_out, bias=bias).to(device=device, dtype=dtype)
        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, edge_scalars, edge_mask):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        edge_scalars : pytorch tensor
            The information to run on every edge.  Expected to be of
            size b x N x N x num_channels_in x 2, where b is the batch size,
            N is the max number of atoms, and the 2 is due to complex
            arithmetic.
        edge_mask : pytorch tensor of bits.
            Masking tensor: 1 if a nonzero output should be returned for that
            edge and zero otherwise.  Depending on the application this may not
            always be the same as edge_mask used in the cormorant class.

        Returns
        -------
        prediction : pytorch tensor
            Predicted values on the edges.  Of size b x N x N x num_out.
        """
        # Reshape scalars appropriately
        es = edge_scalars.shape
        # edge_scalars = edge_scalars.view(es[0:3] + (self.num_channels_in * 2,))

        # First MLP applied to each atom
        x = self.lin(edge_scalars)

        # Zero out masked elements.
        # predict = x * edge_mask.unsqueeze(dim=-1).float()
        prediction = torch.where(edge_mask.unsqueeze(dim=-1), x, self.zero)
        return prediction


class OutputEdgeMLP(nn.Module):
    """
    Runs an MLP on the values on each edge, without communication.

    Parameters
    ----------
    num_channels_in : int
        Number of input channels for every edge
    num_out : int
        Number of output channels for every edge
    num_hidden : int
        Number of hidden layers to use
    layer_width : int
        width of the hidden layers.
    activation : str
        Activation function to use for the nonlinearity.
    device : pytorch device object
        Computer architecture the code is being run on.
    dtype : pytorch datatype
        Type of pytorch datatype
    """
    def __init__(self, num_channels_in, num_out=1, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputEdgeMLP, self).__init__()

        self.num_channels_in = num_channels_in

        self.mlp = BasicMLP(2*num_channels_in, num_out, num_hidden=num_hidden,
                            layer_width=layer_width, activation=activation,
                            device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, edge_scalars, edge_mask):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        edge_scalars : pytorch tensor
            The information to run on every edge.  Expected to be of
            size b x N x N x num_channels_in x 2, where b is the batch size,
            N is the max number of atoms, and the 2 is due to complex
            arithmetic.
        edge_mask : pytorch tensor of bits.
            Masking tensor: 1 if a nonzero output should be returned for that
            edge and zero otherwise.  Depending on the application this may not
            always be the same as edge_mask used in the cormorant class.

        Returns
        -------
        prediction : pytorch tensor
            Predicted values on the edges.  Of size b x N x N x num_out.
        """
        # Reshape scalars appropriately
        es = edge_scalars.shape
        edge_scalars = edge_scalars.view(es[0:3] + (self.num_channels_in * 2,))

        # First MLP applied to each atom
        x = self.mlp(edge_scalars)

        # Zero out masked elements.
        # predict = x * edge_mask.unsqueeze(dim=-1).float()
        prediction = torch.where(edge_mask.unsqueeze(dim=-1), x, self.zero)
        return prediction


class OutputMPNN(nn.Module):
    def __init__(self, channels_in, num_levels=1,
                 soft_cut_rad=None, soft_cut_width=None, hard_cut_rad=None, cutoff_type=['learn'],
                 channels_mlp=-1, num_hidden=1, layer_width=256,
                 activation='leakyrelu', basis_set=(3, 3),
                 device=torch.device('cpu'), dtype=torch.float):
        super(OutputMPNN, self).__init__()

        self.soft_cut_rad = soft_cut_rad
        self.soft_cut_width = soft_cut_width
        self.hard_cut_rad = hard_cut_rad

        if channels_mlp < 0:
            channels_mlp = channels_in

        # List of channels at each level. The factor of two accounts for
        # the fact that the passed messages are concatenated with the input states.
        channels_lvls = [channels_in] + [channels_mlp]*(num_levels-1) + [1]

        self.channels_in = channels_in
        self.channels_mlp = channels_mlp
        self.channels_out = channels_out

        # Set up MLPs
        self.mlps = nn.ModuleList()
        self.masks = nn.ModuleList()
        self.rad_filts = nn.ModuleList()

        for chan_in, chan_out in zip(channels_lvls[:-1], channels_lvls[1:]):
            rad_filt = RadPolyTrig(0, basis_set, chan_in, mix='real', device=device, dtype=dtype)
            mask = MaskLevel(chan_in, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type, device=device, dtype=dtype)
            mlp = BasicMLP(2*chan_in, chan_out, num_hidden=num_hidden, layer_width=layer_width, device=device, dtype=dtype)

            self.mlps.append(mlp)
            self.masks.append(mask)
            self.rad_filts.append(rad_filt)

        self.dtype = dtype
        self.device = device

    def forward(self, features, atom_mask, edge_mask, norms):
        # Unsqueeze the atom mask to match the appropriate dimensions later
        atom_mask = atom_mask.unsqueeze(-1)

        # Get the shape of the input to reshape at the end
        s = features.shape

        # Loop over MPNN levels. There is no "edge network" here.
        # Instead, there is just masked radial functions, that take
        # the role of the adjacency matrix.
        for mlp, rad_filt, mask in zip(self.mlps, self.rad_filts, self.masks):
            # Construct the learnable radial functions
            rad = rad_filt(norms, edge_mask)
            # Convert to a form that MaskLevel expects
            rad[0] = rad[0].unsqueeze(-1)

            # Mask the position function if desired
            edge = mask(rad, edge_mask, norms)
            # Convert to a form that MatMul expects
            edge = edge[0].squeeze(-1)

            # Now pass messages using matrix multiplication with the edge features
            # Einsum b: batch, a: atom, c: channel, x: to be summed over
            features_mp = torch.einsum('baxc,bxc->bac', edge, features)

            # Concatenate the passed messages with the original features
            features_mp = torch.cat([features_mp, features], dim=-1)

            # Now apply a masked MLP
            features = mlp(features_mp, mask=atom_mask)

        # The output are the MLP features reshaped into a set of complex numbers.
        out = features.view(s[0:2] + (self.channels_out, 1, 2))

        return out


class OutputAtomMLP(nn.Module):
    """
    Multilayer perceptron.
    """

    def __init__(self, num_scalars, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputAtomMLP, self).__init__()

        self.num_scalars = num_scalars
        self.basic_mlp = BasicMLP(2*num_scalars, 1, num_hidden=num_hidden, layer_width=layer_width, activation=activation, device=device, dtype=dtype)

    def forward(self, scalars, ignore=None):
        # scalars = scalars.sum(1)
        scalars = scalars.view((scalars.shape[0], scalars.shape[1], 2*self.num_scalars))

        predict = self.basic_mlp(scalars)

        predict = predict.squeeze(-1)

        return predict


class OutputMLP(nn.Module):
    """
    Multilayer perceptron.
    """

    def __init__(self, num_scalars, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputMLP, self).__init__()

        self.num_scalars = num_scalars
        self.basic_mlp = BasicMLP(2*num_scalars, 1, num_hidden=num_hidden, layer_width=layer_width, activation=activation, device=device, dtype=dtype)

    def forward(self, scalars, ignore=None):
        scalars = scalars.sum(1)
        scalars = scalars.view((scalars.shape[0], 2*self.num_scalars))

        predict = self.basic_mlp(scalars)

        predict = predict.squeeze(-1)

        return predict
