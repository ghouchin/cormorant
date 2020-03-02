from cormorant.ase import ASEInterface
from ase.io import read

atoms=read('test.traj')

included_species=[]
for i in range(len(atoms)):
    included_species.append(atoms.numbers[i])

calc=ASEInterface.load('model/LiNMC.pt',num_species=5,included_species=included_species)

atoms=read('/home/ghouchin/ClusterExpansion/cifs/LiNiO2.cif')

atoms.set_calculator(calc)

atoms.get_potential_energy()
