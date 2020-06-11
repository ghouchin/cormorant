from cormorant.ase import ASEInterface
from ase import Atoms
from ase.db import connect

db=connect('molecules.db')


calc=ASEInterface.load('model/molecules.pt')
h=Atoms('H',positions=[[0, 0, 0]])
h2=Atoms('H2',positions=[[0, 0, 0],[0, 0, 70]])

#print(calc.model)

h.set_calculator(calc)
h2.set_calculator(calc)

print('For H')
print(h.get_potential_energy())
print(h2.get_potential_energy())
print(h2.get_potential_energy()/2)


atoms = db.get_atoms(id=200)


cell = atoms.get_cell()
cell[0] = cell[0]*2
atoms.set_cell(cell)

atoms2=atoms.repeat((2,1,1))

atoms.set_calculator(calc)
atoms2.set_calculator(calc)

print('For ',atoms.get_chemical_formula(mode='hill'))
print(atoms.get_potential_energy())
print(atoms2.get_potential_energy())
print(atoms2.get_potential_energy()/2)
