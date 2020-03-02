from ase.calculators.calculator import Calculator
from cormorant.models import CormorantQM9
import torch


class ase_interface(Calculator):
    implemented_properties =['energy', 'forces']
    def __init__(self, model): 
        Calculator.__init__(self)
        self.model = model
        
    @classmethod
    def load(cls, filename):
        saved_run = torch.load(filename)
        args = saved_run['args']
        model = CormorantQM9(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                        args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                        args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                        charge_scale, args.gaussian_mask,
                        args.top, args.input, args.num_mpnn_levels,
                        device=device, dtype=dtype)
        model.load_state_dict(saved_run['model_state'])
        calc = cls(model)
        return calc

    def train(self):
        self.model.train()

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties,system_changes)

        input = convert_atoms_to_input(atoms)
        energy = model.get_energy()
        if 'forces' in properties:
            forces = model.get_forces()


        self.results['forces'] = forces
        self.results['energy'] = energy

