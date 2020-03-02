from ase.calculators.calculator import Calculator
from cormorant.models import CormorantQM9
from cormorant.engine import init_cuda
import torch


class ASEInterface(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model):
        Calculator.__init__(self)
        self.model = model

    @classmethod
    def load(cls, filename, num_species):
        saved_run = torch.load(filename)
        args = saved_run['args']

        # Initialize device and data type
        device, dtype = init_cuda(args)
        model = CormorantQM9(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                             args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                             args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                             args.charge_scale, args.gaussian_mask,
                             args.top, args.input, args.num_mpnn_levels,
                             device=device, dtype=dtype)
        model.load_state_dict(saved_run['model_state'])
        calc = cls(model)
        return calc

    def train(self, trainer):
        trainer.load_checkpoint()
        self.trainer = trainer
        self.trainer.train()

    def calculate(self, atoms, properties, system_changes):
        """
        Populates results dictionary.

        Parameters
        ----------
        atoms : ASE Atoms object
            Atoms object from  ASE
        properties : list of strings
            Properties to calculate.
        system_changes : list
            list of what has changed.     
        """
        Calculator.calculate(self, atoms)

        corm_input = convert_atoms(atoms)
        energy = model(corm_input)
        if 'forces' in properties:
            forces = self._get_forces(energy)
         
        self.results['forces'] = forces
        self.results['energy'] = energy

    def _get_forces(self, energy):
        forces = []
        # Grad must be called for each predicted energy in the batch
        for i, pred in enumerate(energy):
            chunk_forces = -torch.autograd.grad(pred, sample_batch['positions'], create_graph=True, retain_graph=True)[0]
            forces.append(chunk_forces[i])
        return torch.stack(forces, dim=0)



    def convert_atoms(self,atoms):
        data = {}
        atom_charges, atom_positions = [], []
        for i, line in enumerate(datoms.positions):
            atom_charges.append(atoms.numbers[i])
            atom_positions.append(list(line))
        data['charges'] = atom_charges
        data['positions'] = atoms_positions
        data['one_hot'] = self.data['charges'].unsqueeze(-1) == self.included_species.unsqueeze(0).unsqueeze(0)
        return data
