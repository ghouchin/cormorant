import torch
from math import sqrt, pi

from cormorant.cg_lib import CGModule
from cormorant.cg_lib import cg_product

from cormorant.so3_lib import SO3Vec


class SphericalHarmonics(CGModule):
    r"""
    Calculate a list of spherical harmonics :math:`Y^\ell_m(\hat{\bf r})`
    for a :class:`torch.Tensor` of cartesian vectors :math:`{\bf r}`.

    This module subclasses :class:`CGModule`, and maintains similar functionality.

    Parameters
    ----------
    maxl : :class:`int`
        Calculate spherical harmonics from ``l=0, ..., maxl``.
    normalize : :class:`bool`, optional
        Normalize the cartesian vectors used to calculate the spherical harmonics.
    conj : :class:`bool`, optional
        Return the conjugate of the (conventionally defined) spherical harmonics.
    sh_norm : :class:`str`, optional
        Chose the normalization convention for the spherical harmonics.
        The options are:

        - 'qm': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = \frac{2\ell+1}{4\pi}`

        - 'unit': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = 1`
    cg_dict : :class:`CGDict`, optional
        Specify a Clebsch-Gordan Dictionary
    dtype : :class:`torch.torch.dtype`, optional
        Specify the dtype to initialize the :class:`CGDict`/:class:`CGModule` to
    device : :class:`torch.torch.device`, optional
        Specify the device to initialize the :class:`CGDict`/:class:`CGModule` to
    """

    def __init__(self, maxl, normalize=True, conj=False, sh_norm='unit',
                 cg_dict=None, dtype=None, device=None):

        self.normalize = normalize
        self.sh_norm = sh_norm
        self.conj = conj

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

    def forward(self, pos):
        r"""
        Calculate the Spherical Harmonics for a set of cartesian position vectors.

        Parameters
        ----------
        pos : :class:`torch.Tensor`
            Input tensor of cartesian vectors

        Returns
        -------
        sph_harms : :class:`list` of :class:`torch.Tensor`
            Output list of spherical harmonics from :math:`\ell=0` to :math:`\ell=maxl`
        """
        return spherical_harmonics(self.cg_dict, pos, self.maxl,
                                   self.normalize, self.conj, self.sh_norm)


class SphericalHarmonicsRelDebug(CGModule):
    r"""
    Calculate a matrix of spherical harmonics

    .. math::
        \Upsilon_{ij} = Y^\ell_m(\hat{\bf r}_{ij})

    based upon the difference

    .. math::
        {\bf r}_{ij} = {\bf r}^{(1)}_i - {\bf r}^{(2)}_j.

    in two lists of cartesian vectors  :math:`{\bf r}^{(1)}_i`
    and :math:`{\bf r}^{(2)}_j`.


    This module subclasses :class:`CGModule`, and maintains similar functionality.

    Parameters
    ----------
    maxl : :class:`int`
        Calculate spherical harmonics from ``l=0, ..., maxl``.
    normalize : :class:`bool`, optional
        Normalize the cartesian vectors used to calculate the spherical harmonics.
    conj : :class:`bool`, optional
        Return the conjugate of the (conventionally defined) spherical harmonics.
    sh_norm : :class:`str`, optional
        Chose the normalization convention for the spherical harmonics.
        The options are:

        - 'qm': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = \frac{2\ell+1}{4\pi}`

        - 'unit': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = 1`
    cg_dict : :class:`CGDict` or None, optional
        Specify a Clebsch-Gordan Dictionary
    dtype : :class:`torch.torch.dtype`, optional
        Specify the dtype to initialize the :class:`CGDict`/:class:`CGModule` to
    device : :class:`torch.torch.device`, optional
        Specify the device to initialize the :class:`CGDict`/:class:`CGModule` to
    """

    def __init__(self, maxl, normalize=False, conj=False, sh_norm='unit',
                 cg_dict=None, dtype=None, device=None):

        self.normalize = normalize
        self.sh_norm = sh_norm
        self.conj = conj

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

    def forward(self, pos, neighborlist):
        r"""
        Calculate the Spherical Harmonics for a matrix of differences of cartesian
        position vectors `pos` and `neighborlist`.

        Note that `pos1` and `pos2` must agree in every dimension except for
        the second-to-last one.

        Parameters
        ----------
        pos1 : :class:`torch.Tensor`
            First tensor of cartesian vectors :math:`{\bf r}^{(1)}_i`.
        pos2 : :class:`torch.Tensor`
            Second tensor of cartesian vectors :math:`{\bf r}^{(2)}_j`.

        Returns
        -------
        sph_harms : :class:`list` of :class:`torch.Tensor`
            Output matrix of spherical harmonics from :math:`\ell=0` to :math:`\ell=maxl`
        """
        
        '''
        neighbor_pos = []
        num_atoms = pos.shape[1]
        import pdb
        pdb.set_trace()
        neigbor_pos=[]
        for b, batch in enumerate(neighborlist):
            print(b)
            neighbor_pos_batch = []
            for i in range(num_atoms):
                neighbor_pos_i = [[] for i in range(num_atoms)] 
                for image in batch[i]:
                    j=int(image[0])
                    neighbor_pos_i[j].append(pos[b,j]+image[1:])

                for j in [item for item in range(num_atoms) if item not in set(batch[i,:,0].to(dtype=int))]: #list of indices that are not in the neighbor of i
                    neighbor_pos_i[j].append(pos[b,j])
                neighbor_pos_i = [torch.stack(row) for row in neighbor_pos_i]
                neighbor_pos_i= torch.nn.utils.rnn.pad_sequence(neighbor_pos_i, batch_first=True, padding_value=0)

                neighbor_pos_batch.append(neighbor_pos_i)
        
            max_neighbors = max([p.shape[-2] for p in neighbor_pos_batch])
            max_shape = (num_atoms, num_atoms, max_neighbors, 3)
            padded_neighbor_pos_batch = torch.zeros(max_shape, dtype=neighbor_pos_batch[0].dtype, device=neighbor_pos_batch[0].device)
            for idx, prop in enumerate(neighbor_pos_batch):
                s = prop.shape
                padded_neighbor_pos_batch[idx, :, :s[-2], :s[-1]] = prop

            #neighbor_pos_batch = torch.nn.utils.rnn.pad_sequence(neighbor_pos_batch)
            neighbor_pos.append(padded_neighbor_pos_batch)
        
        from cormorant.data.collate import batch_stack
        neighbor_pos = batch_stack(neighbor_pos, edge_mat=True)
        #max_atoms = max([len(p) for p in neighbor_pos])
        #max_neighbors = max([p.shape[-2] for p in neighbor_pos])
        #max_shape = (len(neighbor_pos), max_atoms, mmax_neighbors, 3)
        #padded_neighbor_pos = torch.zeros(max_shape, dtype=neighbor_pos[0].dtype, device=neighbor_pos[0].device)

        #for idx, prop in enumerate(neighbor_pos):
        #    s = prop.shape
        #    padded_neighbor_pos[idx, :s[-2], :s[-1]] = prop
        '''

        return spherical_harmonics_rel_debug(self.cg_dict, pos, neighbor_pos, self.maxl,
                                       self.normalize, self.conj, self.sh_norm)


def spherical_harmonics(cg_dict, pos, maxsh, normalize=True, conj=False, sh_norm='unit'):
    r"""
    Functional form of the Spherical Harmonics. See documentation of
    :class:`SphericalHarmonics` for details.
    """
    s = pos.shape[:-1]

    pos = pos.view(-1, 3)

    if normalize:
        norm = pos.norm(dim=-1, keepdim=True)
        mask = (norm > 0)
        # pos /= norm
        # pos[pos == inf] = 0
        pos = torch.where(mask, pos / norm, torch.zeros_like(pos))

    psi0 = torch.full(s + (1,), sqrt(1/(4*pi)), dtype=pos.dtype, device=pos.device)
    psi0 = torch.stack([psi0, torch.zeros_like(psi0)], -1)
    psi0 = psi0.view(-1, 1, 1, 2)

    sph_harms = [psi0]
    if maxsh >= 1:
        psi1 = pos_to_rep(pos, conj=conj)
        psi1 *= sqrt(3/(4*pi))
        sph_harms.append(psi1)

    if maxsh >= 2:
        new_psi = psi1
        for l in range(2, maxsh+1):
            new_psi = cg_product(cg_dict, [new_psi], [psi1], minl=0, maxl=l, ignore_check=True)[-1]
            # Use equation Y^{m1}_{l1} \otimes Y^{m2}_{l2} = \sqrt((2*l1+1)(2*l2+1)/4*\pi*(2*l3+1)) <l1 0 l2 0|l3 0> <l1 m1 l2 m2|l3 m3> Y^{m3}_{l3}
            # cg_coeff = CGcoeffs[1*(CGmaxL+1) + l-1][5*(l-1)+1, 3*(l-1)+1] # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            cg_coeff = cg_dict[(1, l-1)][5*(l-1)+1, 3*(l-1)+1]  # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            new_psi *= sqrt((4*pi*(2*l+1))/(3*(2*l-1))) / cg_coeff
            sph_harms.append(new_psi)
    sph_harms = [part.view(s + part.shape[1:]) for part in sph_harms]

    if sh_norm == 'qm':
        pass
    elif sh_norm == 'unit':
        sph_harms = [part*sqrt((4*pi)/(2*ell+1)) for ell, part in enumerate(sph_harms)]
    else:
        raise ValueError('Incorrect choice of spherial harmonic normalization!')

    return SO3Vec(sph_harms)


def spherical_harmonics_rel_debug(cg_dict, pos1, pos2, maxsh, normalize=True, conj=False, sh_norm='unit'):
    r"""
    Functional form of the relative Spherical Harmonics. See documentation of
    :class:`SphericalHarmonicsRel` for details.
    """
    rel_pos = pos1.unsqueeze(-2).repeat(1,1,pos2.shape[-2],1).unsqueeze(-3) - pos2
    #rel_pos = pos1.unsqueeze(-2) - pos2.unsqueeze(-3)
    rel_norms = rel_pos.norm(dim=-1, keepdim=True)
    rel_sq_norms = rel_pos.pow(2).sum(dim=-1, keepdim=True)


    #at some point i need a mask for which images are padded and which are real
    rel_sph_harm = spherical_harmonics_add(cg_dict, rel_pos, maxsh, normalize=normalize,
                                       conj=conj, sh_norm=sh_norm)

    return rel_sph_harm, rel_norms.squeeze(-1), rel_sq_norms.squeeze(-1)


def pos_to_rep(pos, conj=False):
    r"""
    Convert a tensor of cartesian position vectors to an l=1 spherical tensor.

    Parameters
    ----------
    pos : :class:`torch.Tensor`
        A set of input cartesian vectors. Can have arbitrary batch dimensions
         as long as the last dimension has length three, for x, y, z.
    conj : :class:`bool`, optional
        Return the complex conjugated representation.


    Returns
    -------
    psi1 : :class:`torch.Tensor`
        The input cartesian vectors converted to a l=1 spherical tensor.

    """
    pos_x, pos_y, pos_z = pos.unbind(-1)

    # Only the y coordinates get mapped to imaginary terms
    if conj:
        pos_m = torch.stack([pos_x, pos_y], -1)/sqrt(2.)
        pos_p = torch.stack([-pos_x, pos_y], -1)/sqrt(2.)
    else:
        pos_m = torch.stack([pos_x, -pos_y], -1)/sqrt(2.)
        pos_p = torch.stack([-pos_x, -pos_y], -1)/sqrt(2.)
    pos_0 = torch.stack([pos_z, torch.zeros_like(pos_z)], -1)

    psi1 = torch.stack([pos_m, pos_0, pos_p], dim=-2).unsqueeze(-3)

    return psi1


def rep_to_pos(rep):
    r"""
    Convert a tensor of l=1 spherical tensors to cartesian position vectors.

    Warning
    -------
    The input spherical tensor must satisfy :math:`F_{-m} = (-1)^m F_{m}^*`,
    so the output cartesian tensor is explicitly real. If this is not satisfied
    an error will be thrown.

    Parameters
    ----------
    rep : :class:`torch.Tensor`
        A set of input l=1 spherical tensors.
        Can have arbitrary batch dimensions as long
        as the last dimension has length three, for m = -1, 0, +1.

    Returns
    -------
    pos : :class:`torch.Tensor`
        The input l=1 spherical tensors converted to cartesian vectors.

    """
    rep_m, rep_0, rep_p = rep.unbind(-2)

    pos_x = (-rep_p + rep_m)/sqrt(2.)
    pos_y = (-rep_p - rep_m)/sqrt(2.)
    pos_z = rep_0

    imag_part = [pos_x[..., 1].abs().mean(), pos_y[..., 0].abs().mean(), pos_z[..., 1].abs().mean()]
    if (any(p > 1e-6 for p in imag_part)):
        raise ValueError('Imaginary part not zero! {}'.format(imag_part))

    pos = torch.stack([pos_x[..., 0], pos_y[..., 1], pos_z[..., 0]], dim=-1)

    return pos


def rep_to_twotensor(rep):
    rep0, rep1_m1, rep1_0, rep1_p1, rep2_m2, rep2_m1, rep2_0, rep2_p1, rep2_p2 = rep.unbind(-2)

    r2 = sqrt(2)
    r3 = sqrt(3)
    r6 = r2 * r3
    xx = -rep0[..., 0]/r3 + rep2_m2[..., 0]/2 - rep2_0[..., 0]/r6 + rep2_p2[..., 0]/2
    xy = rep1_0[..., 1]/r2 + rep2_m2[..., 1]/2 + rep2_p2[..., 1]/2
    xz = -rep1_m1[..., 0]/2 - rep1_p1[..., 0]/2 + rep2_m1[..., 0]/2 - rep2_p1[..., 0]/2
    yx = -rep1_0[..., 1]/r2 + rep2_m2[..., 1]/2 + rep2_p2[..., 1]/2
    yy = -rep0[..., 0]/r3 - rep2_m2[..., 0]/2 - rep2_0[..., 0]/r6 - rep2_p2[..., 0]/2
    yz = -rep1_m1[..., 1]/2 - rep1_p1[..., 1]/2 + rep2_m1[..., 1]/2 - rep2_p1[..., 1]/2

    zx = rep1_m1[..., 0]/2 + rep1_p1[..., 0]/2 + rep2_m1[..., 0]/2 - rep2_p1[..., 0]/2
    zy = rep1_m1[..., 1]/2 + rep1_p1[..., 1]/2 + rep2_m1[..., 1]/2 - rep2_p1[..., 1]/2
    zz = -rep0[..., 0]/r3 + 2*rep2_0[..., 0]/r6

    two_tensor = torch.stack([xx, xy, xz, yx, yy, yz, zx, zy, zz], dim=-1)
    return two_tensor


def twotensor_to_rep(twotensor):
    xx, xy, xz, yx, yy, yz, zx, zy, zz = twotensor.unbind(-1)
    r2 = sqrt(2)
    r3 = sqrt(3)
    r6 = r2 * r3

    rep0_real = (-xx - yy - zz) / r3
    rep0 = torch.stack([rep0_real, torch.zeros_like(rep0_real)], -1)
    rep1_m1_real = -xz + zx
    rep1_m1_imag = yz - zy
    rep1_m1 = torch.stack([rep1_m1_real, rep1_m1_imag], -1)/2
    rep1_0_imag = (-xy + yx) / r2
    rep1_0 = torch.stack([torch.zeros_like(rep1_0_imag), rep1_0_imag], -1)
    rep1_p1 = rep1_m1

    rep2_m2_real = xx - yy
    rep2_m2_imag = - xy - yx
    rep2_m2 = torch.stack([rep2_m2_real, rep2_m2_imag], -1) / 2
    rep2_m1_real = xz + zx
    rep2_m1_imag = -yx - zy
    rep2_m1 = torch.stack([rep2_m1_real, rep2_m1_imag], -1) / 2
    rep2_0 = -xx - yy + 2 * zz
    rep2_0 = torch.stack([rep2_0, torch.zeros_like(rep2_0)], -1) / r6
    rep2_p1 = torch.stack([rep2_m1_real, rep2_m1_imag], -1) / 2
    rep2_p2 = rep2_m2

    rep1_all = torch.stack([rep1_m1, rep1_0, rep1_p1], dim=-2)
    rep1_all = rep1_all.unsqueeze(-3)
    rep2_all = torch.stack([rep2_m2, rep2_m1, rep2_0, rep2_p1, rep2_p2], dim=-2)
    rep2_all = rep2_all.unsqueeze(-3)
    return rep0, rep1_all, rep2_all
