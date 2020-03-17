from cormorant.cg_lib import CGProduct, CGModule
from cormorant.nn import MaskLevel
from cormorant.nn import CatMixReps, DotMatrix


class CormorantEdgeLevel(CGModule):
    """
    Scalar edge level as part of the Clebsch-Gordan layers are described in the
    Cormorant paper.

    The input takes in an edge network from a previous level,
    the representations from the previous level, and also a set of
    scalar functions of the relative positions between all atoms.

    Parameters
    ----------
    tau_atom : :class:`SO3Tau`
        Multiplicity (tau) of the input atom representations
    tau_edge : :class:`SO3Tau`
        Multiplicity (tau) of the input edge layer representations
    tau_pos : :class:`SO3Tau`
        Multiplicity (tau) of the input set of position functions
    nout : :obj:`int`
        Number of output channels to mix the concatenated scalars to.
    max_sh : :obj:`int`
        Maximum weight of the spherical harmonics.
    cutoff_type : :obj:`str`
        `cutoff_type` to be passed to :class:`cormorant.nn.MaskLevel`
    hard_cut_rad : :obj:`float`
        `hard_cut_rad` to be passed to :class:`cormorant.nn.MaskLevel`
    soft_cut_rad : :obj:`float`
        `soft_cut_rad` to be passed to :class:`cormorant.nn.MaskLevel`
    soft_cut_width : :obj:`float`
        `soft_cut_width` to be passed to :class:`cormorant.nn.MaskLevel`
    gaussian_mask : :obj:`bool`
        `gaussian_mask` to be passed to :class:`cormorant.nn.MaskLevel`
    cat : :obj:`bool`
        Concatenate all the scalars in :class:`cormorant.nn.DotMatrix`
    device : :obj:`torch.device`
        Device to initialize the level to
    dtype : :obj:`torch.dtype`
        Data type to initialize the level to
    """

    def __init__(self, tau_atom, tau_edge, tau_pos, nout, max_sh,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False, device=None, dtype=None,
                 use_edge_in=True, use_edge_dot=True, use_pos_funcs=True):
        super().__init__(device=device, dtype=dtype)
        device, dtype = self.device, self.dtype

        # Set up type of edge network depending on specified input operations
        self.dot_matrix = DotMatrix(tau_atom, cat=cat,
                                    device=self.device, dtype=self.dtype)
        tau_dot = self.dot_matrix.tau

        # Set up mixing layer
        edge_taus = []
        if use_edge_in:
            edge_taus.append(tau_edge)
        if use_edge_dot:
            edge_taus.append(tau_dot)
        if use_pos_funcs:
            edge_taus.append(tau_pos)

        edge_taus = [tau for tau in edge_taus if tau is not None]
        self.cat_mix = CatMixReps(edge_taus, nout, real=False, maxl=max_sh,
                                  device=self.device, dtype=self.dtype)
        self.tau = self.cat_mix.tau

        # Set up edge mask layer
        self.mask_layer = MaskLevel(nout, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type,
                                    gaussian_mask=gaussian_mask, device=self.device, dtype=self.dtype)
        self.use_edge_in = use_edge_in
        self.use_edge_dot = use_edge_dot
        self.use_pos_funcs = use_pos_funcs

    def forward(self, edge_in, atom_reps, pos_funcs, base_mask, norms, sq_norms):
        # Concatenate and mix the different types of edge features together
        edge_features = []
        if self.use_edge_in:
            edge_features.append(edge_in)
        if self.use_edge_dot:
            # Calculate the dot product matrix.
            edge_dot = self.dot_matrix(atom_reps)
            edge_features.append(edge_dot)
        if self.use_pos_funcs:
            edge_features.append(pos_funcs)

        edge_mix = self.cat_mix(edge_features)

        # Apply mask to layer -- For now, only can be done after mixing.
        edge_net = self.mask_layer(edge_mix, base_mask, norms, sq_norms)

        return edge_net


class CormorantAtomLevel(CGModule):
    """
    Atom level as part of the Clebsch-Gordan layers are described in the
    Cormorant paper.

    The input takes in an edge network from a previous level, along
    with a set of representations that correspond to edges between atoms.
    Applies a masked Clebsh-Gordan operation.

    Parameters
    ----------
    tau_in : :class:`SO3Tau`
        Multiplicity (tau) of the input atom representations
    tau_pos : :class:`SO3Tau`
        Multiplicity (tau) of the input set of position functions
    maxl : :obj:`int`
        Maximum weight of the spherical harmonics.
    num_channels : :obj:`int`
        Number of output channels to mix the concatenated :class:`SO3Vec` to.
    weight_init : :obj:`str`
        Weight initialization function.
    level_gain : :obj:`int`
        Gain for the weights at each level.

    device : :obj:`torch.device`
        Device to initialize the level to
    dtype : :obj:`torch.dtype`
        Data type to initialize the level to
    cg_dict : :obj:`cormorant.cg_lib.CGDict`
        Clebsch-Gordan dictionary for the CG levels.

    """

    def __init__(self, tau_in, tau_pos, maxl, num_channels, level_gain, weight_init,
                 device=None, dtype=None, cg_dict=None,
                 use_ag=True, use_sq=True, use_id=True):
        super().__init__(maxl=maxl, device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.tau_in = tau_in
        self.tau_pos = tau_pos

        # Operations linear in input reps
        catmix_taus = []
        if use_ag:
            # Operations linear in input reps
            self.cg_aggregate = CGProduct(tau_pos, tau_in, maxl=self.maxl, aggregate=True,
                                          device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)
            tau_ag = list(self.cg_aggregate.tau)
            catmix_taus.append(tau_ag)
        if use_id:
            catmix_taus.append(tau_in)
        if use_sq:
            self.cg_power = CGProduct(tau_in, tau_in, maxl=self.maxl,
                                      device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)
            tau_sq = list(self.cg_power.tau)
            catmix_taus.append(tau_sq)

        self.cg_power = CGProduct(tau_in, tau_in, maxl=self.maxl,
                                  device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)
        tau_sq = list(self.cg_power.tau)

        self.cat_mix = CatMixReps(catmix_taus, num_channels,
                                  maxl=self.maxl, weight_init=weight_init, gain=level_gain,
                                  device=self.device, dtype=self.dtype)
        self.tau = self.cat_mix.tau
        self.use_ag = use_ag
        self.use_sq = use_sq
        self.use_id = use_id

    def forward(self, atom_reps, edge_reps, mask):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        atom_reps : SO3Vec
            Representation of the atomic environment.
        edge_reps : SO3Vec
            Representation of the connections between atoms
        mask : pytorch Tensor
            Mask determining which elements of atom_reps are active.

        Returns
        -------
        reps_out : SO3Vec
            Output representation of the atomic environment.
        """
        atom_features = []
        if self.use_ag:
            # Aggregate information based upon edge reps
            reps_ag = self.cg_aggregate(edge_reps, atom_reps)
            atom_features.append(reps_ag)

        if self.use_id:
            atom_features.append(atom_reps)

        if self.use_sq:
            # CG non-linearity for each atom
            reps_sq = self.cg_power(atom_reps, atom_reps)
            atom_features.append(reps_sq)

        # Concatenate and mix results
        reps_out = self.cat_mix(atom_features)

        return reps_out
