import numpy as np
from cormorant.ase.ase_interface_debug import ASEInterfaceDebug
from ase.io import read
#from pytorch_memlab import profile


#@profile
def main():
    calc = ASEInterfaceDebug()
    import pdb
    pdb.set_trace()
    calc.train('small.db',force_factor=0, num_epoch=10, batch_size=30, num_channels=3)




    atoms = read('test.traj')
    e_DFT=atoms.get_potential_energy()
    f_DFT=atoms.get_forces()
    print(e_DFT)
    print(f_DFT)
    
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


if __name__ == '__main__':
    main()