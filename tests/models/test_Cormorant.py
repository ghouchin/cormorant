import pytest

from cormorant.models import CormorantQM9, CormorantMD17, CormorantASE


class TestCormorant():
    @pytest.mark.parametrize('Cormorant', [CormorantQM9, CormorantMD17])
    @pytest.mark.parametrize('maxl', [1, 2, [1, 2]])
    @pytest.mark.parametrize('max_sh', [1, 2])
    @pytest.mark.parametrize('num_channels', [1, 5, [1, 2], [2, 1, 4]])
    @pytest.mark.parametrize('cutoff_type', ['hard', 'soft', 'learn', ['hard', 'soft'], ['hard', 'learn']])
    @pytest.mark.parametrize('level_gain', [1, 10])
    def test_Cormorant(self, Cormorant, maxl, max_sh, num_channels, level_gain, cutoff_type, sample_batch):
        data, num_species, charge_scale = sample_batch

        num_cg_levels = 3
        hard_cut_rad = 1.
        soft_cut_rad = 1.
        soft_cut_width = 1.
        weight_init = 'rand'
        charge_power = 2
        basis_set = (3, 3)
        gaussian_mask = False
        top = 'linear'
        input = 'linear'
        num_mpnn_layers = 2

        # First test initialization
        cormorant = Cormorant(maxl, max_sh, num_cg_levels, num_channels, num_species,
                              cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                              weight_init, level_gain, charge_power, basis_set,
                              charge_scale, gaussian_mask,
                              top, input, num_mpnn_layers)

        # Get data and then try pushing through a single example
        cormorant(data)


class TestASECormorant():
    @pytest.mark.parametrize('cutoff_type', ['hard', 'soft', 'learn', ['hard', 'soft'], ['hard', 'learn']])
    def test_Cormorant(self, cutoff_type, sample_batch):
        num_channels = 4
        level_gain = 10
        maxl = max_sh = 2
        data, num_species, charge_scale = sample_batch
        data['relative_pos'] = data['positions'].unsqueeze(-2) - data['positions'].unsqueeze(-3)

        num_cg_levels = 3
        hard_cut_rad = 1.
        soft_cut_rad = 1.
        soft_cut_width = 1.
        weight_init = 'rand'
        charge_power = 2
        basis_set = (3, 3)
        gaussian_mask = False
        top = 'linear'
        input = 'linear'
        num_mpnn_layers = 2

        # First test initialization
        cormorant = CormorantASE(maxl, max_sh, num_cg_levels, num_channels, num_species,
                                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                                 weight_init, level_gain, charge_power, basis_set,
                                 charge_scale, gaussian_mask,
                                 top, input, num_mpnn_layers)

        # Get data and then try pushing through a single example
        cormorant(data)
