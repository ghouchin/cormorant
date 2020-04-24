import torch


def batch_stack(props, edge_mat=False):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack
    edge_mat : bool
        The included tensor refers to edge properties, and therefore needs
        to be stacked/padded along two axes instead of just one.

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        props = [torch.tensor(pi) for pi in props]
    if props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_atoms = max([len(p) for p in props])
        if props[0].dim()>2:
            max_pad = max([p.shape[2] for p in props])   
            max_shape = (len(props), max_atoms, max_atoms, max_pad) + props[0].shape[3:]
            padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)
            for idx, prop in enumerate(props):
                this_atoms = len(prop)
                this_pad = prop.shape[2]
                padded_tensor[idx, :this_atoms, :this_atoms, :this_pad] = prop
        else:
            max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]    
            padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

            for idx, prop in enumerate(props):
                this_atoms = len(prop)
                padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor

def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


def collate_fn(batch, edge_features=None):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.
    edge_features : list or None
        Name of the features that transform as two-tensors.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    if edge_features is None:
        edge_features = []
    collated_batch = {}
    for prop in batch[0].keys():
        in_edge_features = (prop in edge_features)
        collated_batch[prop] = batch_stack([mol[prop] for mol in batch], in_edge_features)
    batch = collated_batch

    to_keep = (batch['charges'].sum(0) > 0)

    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
    if 'relative_pos' in batch.keys():
        batch['relative_pos'] = batch['relative_pos'][:, :, to_keep, ...]
    if 'neighbor_pos' in batch.keys():
        batch['neighbor_pos'] = batch['neighbor_pos'][:, :, to_keep, ...]

    atom_mask = batch['charges'] > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    batch['atom_mask'] = atom_mask
    batch['edge_mask'] = edge_mask

    return batch
