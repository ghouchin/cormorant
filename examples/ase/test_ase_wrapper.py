import numpy as np
from cormorant.ase import ASEInterface
from ase.io import read

calc = ASEInterface()
calc.train('test_force.db',force_factor=1., num_epoch=1, batch_size=1)



atoms = read('test.traj')
e_DFT=atoms.get_potential_energy()
f_DFT=atoms.get_forces()
#print(e_DFT)
#print(f_DFT)

#included_species = []
#for i in range(len(atoms)):
#    included_species.append(atoms.numbers[i])
#included_species = list(np.unique(included_species))

#calc = ASEInterface.load('../../model/LiNMC.pt', num_species=5, included_species=included_species)

# atoms=read('/home/ghouchin/ClusterExpansion/cifs/LiNiO2.cif')

atoms.set_calculator(calc)

print(atoms.get_potential_energy())
print(atoms.get_forces())
print("Done!")
