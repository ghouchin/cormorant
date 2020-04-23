import logging
import os
import torch
import tarfile
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ase.neighborlist import mic
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from cormorant.data.collate import batch_stack


charge_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
               'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
               'K': 19, 'Ca': 20, 'Ti': 22, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Pd': 46}


def split_dataset(data, split_idxs):
    """
    Splits a dataset according to the indices given.

    Parameters
    ----------
    data : dict
        Dictionary to split.
    split_idxs :  dict
        Dictionary defining the split.  Keys are the name of the split, and
        values are the keys for the items in data that go into the split.

    Returns
    -------
    split_dataset : dict
        The split dataset.
    """
    split_data = {}
    for set, split in split_idxs.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data

# def save_database()


def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules


def process_xyz_md17(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the MD-17 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    line_counter = 0
    atom_positions = []
    atom_types = []
    for line in xyz_lines:
        if line[0] == '#':
            continue
        if line_counter == 0:
            num_atoms = int(line)
        elif line_counter == 1:
            split = line.split(';')
            assert (len(split) == 1 or len(split) == 2), 'Improperly formatted energy/force line.'
            if (len(split) == 1):
                e = split[0]
                f = None
            elif (len(split) == 2):
                e, f = split
                f = f.split('],[')
                atom_energy = float(e)
                atom_forces = [[float(x.strip('[]\n')) for x in force.split(',')] for force in f]
        else:
            split = line.split()
            if len(split) == 4:
                type, x, y, z = split
                atom_types.append(split[0])
                atom_positions.append([float(x) for x in split[1:]])
            else:
                logging.debug(line)
        line_counter += 1

    atom_charges = [charge_dict[type] for type in atom_types]

    molecule = {'num_atoms': num_atoms, 'energy': atom_energy, 'charges': atom_charges,
                'forces': atom_forces, 'positions': atom_positions}

    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def process_ase(data, process_file_fn, file_ext=None, file_idx_list=None, force_train=False):
    """
    Takes an ase database and apply a predefined data processing script to each
    entry. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to the ASE database.
    process_file_fn : callable
        Function to process atoms. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : array?????
        This is the list of the indexes in the split of the database you are processing.
    stack : bool, optional
        ?????
    """
    from ase.db import connect
    # from ase.io import read
    logging.info('Processing ASE database file: {}'.format(data))

    molecules = []

    with connect(data) as db:
        for id in file_idx_list:
            atoms = db.get_atoms(id=int(id),attach_calculator=False)
            molecules.append(process_file_fn(atoms, force_train))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    #need to pad and stack if saving 
    molecules = {key: batch_stack(val, edge_mat=not bool(3-len(val[0].shape)) ) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}
    #molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}
    # If stacking is desireable, pad and then stack.
    # if stack:
    #     molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules


def process_db_row(data, forcetrain=False):
    """
    Read an ase-db row and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    """
    molecule = _process_structure(data)

    # prop_strings = ['energy', 'forces', 'dipole', 'initial_magmoms']
    if forcetrain:
        prop_strings = ['energy', 'forces']
        mol_props = [torch.Tensor([data.get_potential_energy()]), torch.from_numpy(data.get_forces())]
    else:
        prop_strings = ['energy']
        mol_props = [torch.Tensor([data.get_potential_energy()])]

    mol_props = dict(zip(prop_strings, mol_props))
    molecule.update(mol_props)
    return molecule


def _process_structure(data):
    #num_atoms = data.natoms
    #num_atoms = data.get_number_of_atoms()
    num_atoms = data.get_global_number_of_atoms()

    # atom_charges, atom_positions, rel_positions = [], [], []
    # for i, ri in enumerate(data.positions):
    #     atom_charges.append(data.numbers[i])
    #     atom_positions.append(list(ri))
    # atom_positions = np.array(atom_positions)

    # for ri in data.positions:
    #     rel_pos = []
    #     for rj in data.positions:
    #         rij = np.array(ri)-np.array(rj)
    #         rel_pos.append(list(mic(rij, data.cell)))
    #     rel_positions.append(rel_pos)
    rel_positions = np.expand_dims(data.positions, axis=-2) - np.expand_dims(data.positions, axis=-3)
    #rel_positions = np.array([mic(atoms_pos, data.cell) for atoms_pos in rel_positions])    
    rel_positions = torch.from_numpy(rel_positions)
    atom_positions = torch.from_numpy(data.positions)
    atom_charges = torch.from_numpy(data.numbers).float()
    

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions, 'relative_pos': rel_positions}
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}
    return molecule

def process_db_row_debug(data, forcetrain=False, cutoff=None):
    """
    Read an ase-db row and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type.

    Parameters
    ----------
    data : ASE atoms object
    cutoff : float 
        hard_cutoff for calculating the neighborlist

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    """
    molecule = _process_structure_neighborlist(data, cutoff)

    # prop_strings = ['energy', 'forces', 'dipole', 'initial_magmoms']
    if forcetrain:
        prop_strings = ['energy', 'forces']
        mol_props = [data.get_potential_energy(), data.get_forces()]
    else:
        prop_strings = ['energy']
        mol_props = [data.get_potential_energy()]

    mol_props = dict(zip(prop_strings, mol_props))
    molecule.update(mol_props)
    return molecule



def _process_structure_neighborlist(data, cutoff):
    num_atoms = data.get_global_number_of_atoms()
    #rel_positions = np.expand_dims(data.positions, axis=-2) - np.expand_dims(data.positions, axis=-3)
    #rel_positions = np.array([mic(atoms_pos, data.cell) for atoms_pos in rel_positions])
    #rel_positions = torch.from_numpy(rel_positions)
    atom_positions = torch.from_numpy(data.positions)
    atom_charges = torch.from_numpy(data.numbers).float()
    nl = NeighborList(cutoffs = [cutoff/2] * num_atoms, self_interaction=False, bothways = True, primitive=NewPrimitiveNeighborList)
    nl.update(data)
    cell = torch.from_numpy(data.cell.array)
    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    '''
    neighborlist = []
    for i in range(num_atoms):
        indices, offsets = nl.get_neighbors(i)
        neighborlist.append(torch.cat((torch.Tensor(indices).unsqueeze(-1).to(dtype=float), torch.mm(torch.Tensor(offsets).to(dtype=float),cell) ), 1))
    neighborlist = torch.nn.utils.rnn.pad_sequence(neighborlist, batch_first=True, padding_value=0) #what is a good value?   
    '''


    
    neighbor_pos = [] #a list of the shifts of neighbors
    neighborlist = torch.zeros([num_atoms, num_atoms], dtype=int) #a list of number of images of j for every i 

    for i in range(num_atoms):
        indices, offsets = nl.get_neighbors(i)
        neighbors = torch.cat((torch.Tensor(indices).unsqueeze(-1), torch.Tensor(offsets)), 1)
        neighbor_pos_i = [[] for i in range(num_atoms)] 
        for image in neighbors:
            j=int(image[0])
            #neighbor_pos_i[j].append(atom_positions[j]+(image[1:]*cell).sum(dim=0))
            neighbor_pos_i[j].append(torch.mm(image[1:],cell))
            neighborlist[i,j] += 1

        for j in [item for item in range(num_atoms) if item not in set(indices)]: #list of indices that are not in the neighbor of i
            #neighbor_pos_i[j].append(atom_positions[j])
            neighbor_pos_i[j].append(torch.zeros([3]))

        neighbor_pos_i = [torch.stack(row) for row in neighbor_pos_i]
        neighbor_pos_i = torch.nn.utils.rnn.pad_sequence(neighbor_pos_i, batch_first=True, padding_value=0)
        neighbor_pos.append(neighbor_pos_i)
    neighbor_pos = torch.nn.utils.rnn.pad_sequence(neighbor_pos, batch_first=True, padding_value=0)




    #for i in range(num_atoms):
    #    neighbor_pos_i = []
    #    indices, offsets = nl.get_neighbors(i)
    #    neighbors = torch.cat((torch.Tensor(indices).unsqueeze(-1), torch.Tensor(offsets)), 1)
    #    for j in range(num_atoms):
    #        neighbor_pos_i_j=[]
    #        for image in neighbors:
    #            if image[0] == j:
    #                neighbor_pos_i_j.append(atom_positions[j]+(image[1:]*cell).sum(dim=0))
    #        if neighbor_pos_i_j==[]:
    #            neighbor_pos_i.append(atom_positions[j].unsqueeze(0))
    #        else:
    #            neighbor_pos_i.append(torch.stack(neighbor_pos_i_j))
    #
    #    neighbor_pos_i = torch.nn.utils.rnn.pad_sequence(neighbor_pos_i, batch_first=True, padding_value=-100)
    #    neighbor_pos.append(neighbor_pos_i)
    #neighbor_pos = torch.nn.utils.rnn.pad_sequence(neighbor_pos, batch_first=True, padding_value=-100)


    #neighborlist = torch.nn.utils.rnn.pad_sequence(neighborlist, batch_first=True, padding_value=-100) #what is a good value? 

    molecule.update({'neighbor_pos': neighbor_pos})
    molecule.update({'neighborlist': neighborlist})

    return molecule
