import pytest
import torch
from cormorant.so3_lib import rotations as rot
from cormorant.so3_lib import SO3Vec, SO3Scalar
from cormorant.nn import RadialFilters
from cormorant.models import CormorantAtomLevel, CormorantEdgeLevel
from cormorant.cg_lib import SphericalHarmonicsRel
# from cormorant.models import CormorantEdgeLevel


# # @pytest.fixture(scope='module')
# def build_environment(tau, maxl, num_channels, level_gain=1, weight_init='rand',
#                       cg_dict=None):
#     datasets, num_species, charge_scale = get_dataloader()
#     data = next(iter(datasets))
#     device, dtype = data['positions'].device, data['positions'].dtype
#     sph_harms = SphericalHarmonicsRel(maxl-1, conj=True, device=device,
#                                       dtype=dtype, cg_dict=cg_dict)
#     return datasets, data, num_species, charge_scale, sph_harms


def prep_input(data, taus, maxl):
    atom_positions = data['positions']
    atom_scalar_list = [torch.randn(atom_positions.shape[:2] + (taus, 2*l+1, 2)) for l in range(maxl)]
    # atom_scalar_list = [torch.randn(atom_positions.shape[:2] + (num_channels, 1, 2))]
    # atom_scalar_list += [torch.zeros(atom_positions.shape[:2] + (num_channels, 2*l+1, 2)) for l in range(1, maxl)]
    atom_scalars = SO3Vec(atom_scalar_list)
    atom_mask = data['atom_mask']
    edge_mask = data['edge_mask']
    edge_scalars = torch.tensor([])
    return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions


class TestCormorantAtomLevel(object):
    # @pytest.mark.parametrize('tau_atom', [1, 3])
    # @pytest.mark.parametrize('tau_edge', [1, 3])
    @pytest.mark.parametrize('tau', [1, 3])
    @pytest.mark.parametrize('num_channels', [3])
    @pytest.mark.parametrize('maxl', [1, 3])
    def test_covariance(self, tau, num_channels, maxl, sample_batch):
        # setup the environment
        # env = build_environment(tau, maxl, num_channels)
        # datasets, data, num_species, charge_scale, sph_harms = env
        data, __, __ = sample_batch
        device, dtype = data['positions'].device, data['positions'].dtype
        sph_harms = SphericalHarmonicsRel(maxl-1, conj=True, device=device,
                                          dtype=dtype, cg_dict=None)
        D, R, __ = rot.gen_rot(maxl, device=device, dtype=dtype)

        # Build Atom layer
        tlist = [tau] * maxl
        print(tlist)
        atom_lvl = CormorantAtomLevel(tlist, tlist, maxl, num_channels, 1, 'rand',
                                      device=device, dtype=dtype, cg_dict=None)

        # Setup Input
        atom_rep, atom_mask, edge_scalars, edge_mask, atom_positions = prep_input(data, tau, maxl)
        atom_positions_rot = rot.rotate_cart_vec(R, atom_positions)

        # Get nonrotated data
        spherical_harmonics, norms, __ = sph_harms(atom_positions, atom_positions)
        edge_rep_list = [torch.cat([sph_l] * tau, axis=-3) for sph_l in spherical_harmonics]
        edge_reps = SO3Vec(edge_rep_list)
        print(edge_reps.shapes)
        print(atom_rep.shapes)

        # Get Rotated output
        output = atom_lvl(atom_rep, edge_reps, atom_mask)
        output = output.apply_wigner(D)

        # Get rotated outputdata
        atom_rep_rot = atom_rep.apply_wigner(D)
        spherical_harmonics_rot, __, __ = sph_harms(atom_positions_rot, atom_positions_rot)
        edge_rep_list_rot = [torch.cat([sph_l] * tau, axis=-3) for sph_l in spherical_harmonics_rot]
        edge_reps_rot = SO3Vec(edge_rep_list_rot)
        output_from_rot = atom_lvl(atom_rep_rot, edge_reps_rot, atom_mask)

        for i in range(maxl):
            assert(torch.max(torch.abs(output_from_rot[i] - output[i])) < 1E-5)


class TestCormorantEdgeLevel(object):
    # @pytest.mark.parametrize('num_channels', [3, 5])
    @pytest.mark.parametrize('num_channels', [7])
    # @pytest.mark.parametrize('maxl', [1, 3])
    @pytest.mark.parametrize('tau', [1, 3])
    # @pytest.mark.parametrize('basis', [1, 3])
    @pytest.mark.parametrize('maxl', [1, 4])
    @pytest.mark.parametrize('basis', [3])
    @pytest.mark.parametrize('edge_net_type', [None, 'rand'])
    # @pytest.mark.parametrize('edge_net_type', [None])
    def test_covariance(self, tau, num_channels, maxl, basis, edge_net_type, sample_batch):
        # env = build_environment(tau, maxl, num_channels)
        # datasets, data, num_species, charge_scale, sph_harms = env
        data, __, __ = sample_batch
        device, dtype = data['positions'].device, data['positions'].dtype
        sph_harms = SphericalHarmonicsRel(maxl-1, conj=True, device=device,
                                          dtype=dtype, cg_dict=None)
        batch_size, natoms = data['positions'].shape[:2]
        D, R, __ = rot.gen_rot(maxl, device=device, dtype=dtype)
        # Setup Input
        atom_reps, atom_mask, edge_scalars, edge_mask, atom_positions = prep_input(data, tau, maxl)
        atom_positions_rot = rot.rotate_cart_vec(R, atom_positions)
        atom_reps_rot = atom_reps.apply_wigner(D)

        # Calculate spherical harmonics and radial functions
        __, norms, sq_norms = sph_harms(atom_positions, atom_positions)
        __, norms_rot, sq_norms = sph_harms(atom_positions_rot, atom_positions_rot)

        rad_funcs = RadialFilters([maxl-1], [basis, basis], [num_channels], 1,
                                  device=device, dtype=dtype)
        rad_func_levels = rad_funcs(norms, edge_mask * (norms > 0))
        tau_pos = rad_funcs.tau[0]

        # Build the initial edge network
        if edge_net_type is None:
            edge_reps = None
        elif edge_net_type == 'rand':
            reps = [torch.randn((batch_size, natoms, natoms, tau, 2)) for i in range(maxl)]
            edge_reps = SO3Scalar(reps)
        else:
            raise ValueError

        # Build Edge layer
        tlist = [tau] * maxl
        tau_atoms = tlist
        tau_edge = tlist
        if edge_net_type is None:
            tau_edge = []

        edge_lvl = CormorantEdgeLevel(tau_atoms, tau_edge, tau_pos, num_channels, maxl,
                                      cutoff_type='soft', device=device, dtype=dtype,
                                      hard_cut_rad=1.73, soft_cut_rad=1.73,
                                      soft_cut_width=0.2)

        output_edge_reps = edge_lvl(edge_reps, atom_reps, rad_func_levels[0], edge_mask, norms, sq_norms)
        output_edge_reps_rot = edge_lvl(edge_reps, atom_reps_rot, rad_func_levels[0], edge_mask, norms, sq_norms)

        for i in range(maxl):
            assert(torch.max(torch.abs(output_edge_reps[i] - output_edge_reps_rot[i])) < 1E-5)
