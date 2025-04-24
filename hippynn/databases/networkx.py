import os
import json

import numpy as np
import torch
import networkx as nx
from networkx.readwrite import json_graph

from .database import Database


def dump_json_graph(G, name, quiet=False):
    """Write graph out in json format
 
    Parameters
    ---------- 
        self : object 
            DFN Class
        
        G :networkX graph
            NetworkX Graph based on the DFN
        
        name : string
             Name of output file (no .json)

    Returns
    -------

    Notes
    -----

"""
    filename = f"{name}.json"
    if not quiet: print(f"--> Dumping Graph into file: {filename} ")
    jsondata = json_graph.node_link_data(G)
    with open(filename, 'w') as fp:
        json.dump(jsondata, fp)
    if not quiet: print("--> Complete")


def load_json_graph(name, quiet=False):
    """ Read in graph from json format

    Parameters
    ---------- 
        self : object 
            DFN Class
        name : string
             Name of input file (no .json)

    Returns
    -------
        G :networkX graph
            NetworkX Graph based on the DFN
"""

    if not quiet: print(f"Loading Graph in file: {name}")
    with open(name) as fp:
        G = json_graph.node_link_graph(json.load(fp))
    if not quiet: print("Complete")
    return G


def to_tensor_with_correct_dtype(x):
    # Handle PyTorch tensor
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            return x.to(dtype=torch.get_default_dtype())
        elif x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return x.to(dtype=int)
        else:
            raise TypeError(f"Unsupported tensor dtype: {x.dtype}")

    # Handle NumPy array
    elif isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.integer):
            return torch.as_tensor(x, dtype=int)
        elif np.issubdtype(x.dtype, np.floating):
            return torch.as_tensor(x, dtype=torch.get_default_dtype())
        else:
            raise TypeError(f"Unsupported NumPy dtype: {x.dtype}")

    # Handle nested lists
    elif isinstance(x, (list, tuple)):
        sample = x[0]
        while isinstance(sample, (list, tuple)):
            sample = sample[0]

        if isinstance(sample, (int, np.int64)):
            return torch.tensor(x, dtype=int)
        elif isinstance(sample, float):
            return torch.tensor(x, dtype=torch.get_default_dtype())
        else:
            raise TypeError(f"Unsupported element type in list: {type(sample)}")

    else:
        raise TypeError(f"Input must be a tensor, NumPy array, or nested list, got: {type(x)}")


def remap_nx_ids_sequential(G):
    # Map old node IDs to new sequential integers
    id_map = {old_id: new_id for new_id, old_id in enumerate(G.nodes)}
    
    # Create new graph of same type
    H = G.__class__()
    H.graph.update(G.graph)  # preserve graph-level attributes
    
    # Add nodes with new IDs and copy node attributes
    for old_id, new_id in id_map.items():
        H.add_node(new_id, **G.nodes[old_id])
    
    # Add edges with new node IDs and copy edge attributes
    for u, v, data in G.edges(data=True):
        H.add_edge(id_map[u], id_map[v], **data)
    
    return H, id_map


def nx_to_npz(graph: nx.Graph):
    # Sort nodes for consistent ordering
    graph, _ = remap_nx_ids_sequential(graph)

    nodes = list(graph.nodes())

    # Collect all node attributes
    node_attr_keys = sorted({key for node in nodes for key in graph.nodes[node]})

    # Build node attribute arrays
    node_data = {'ids': np.array(nodes)}
    for key in node_attr_keys:
        node_data[key] = np.array([
            graph.nodes[node].get(key, None) for node in nodes
        ])

    # Edge data containers
    edge_pairs = []
    edge_attrs = {key: [] for key in {k for _, _, d in graph.edges(data=True) for k in d}}

    for u, v, data in graph.edges(data=True):
        # Add (u, v) and (v, u)
        edge_pairs.extend([(u, v), (v, u)])
        for key in edge_attrs:
            val = data.get(key, None)
            edge_attrs[key].extend([val, val])

    # Final assembly
    edge_data = {'edges': np.array(edge_pairs)}
    for key, values in edge_attrs.items():
        edge_data[key] = np.array(values)

    return node_data, edge_data


def pairs_values_to_sparse(pairs, values=None, max_n_atoms=None):
    # Step 1: Build the indices list
    indices = []
    for frame_idx, pair_list in enumerate(pairs):
        for i, j in pair_list:
            indices.append([frame_idx, i, j])

    # Step 2: Convert to tensor
    indices = to_tensor_with_correct_dtype(indices).T  # shape: [3, N] where N is total number of pairs
    if values is None: 
        values = torch.ones(indices.shape[1], dtype=int)  # All values are 1
    else:
        values = to_tensor_with_correct_dtype(values)
        
    # Step 3: Determine size
    n_frames = len(pairs)
    max_i = max(p[0] for frame in pairs for p in frame)
    max_j = max(p[1] for frame in pairs for p in frame)
    max_n_atoms = max_n_atoms or (max(max_i, max_j) + 1)

    size = (n_frames, max_n_atoms, max_n_atoms)

    # Step 4: Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size)

    return sparse_tensor


def array_list_to_padded_tensor(array_list, padding_value=0):
    max_len = max(arr.shape[0] for arr in array_list)
    
    padded_arrays = []
    for arr in array_list:
        pad_after = max_len - arr.shape[0]
        if pad_after > 0:
            pad_width = ((0, pad_after),) + ((0, 0),) * (arr.ndim - 1)
            padded = np.pad(arr, pad_width, constant_values=padding_value)
        else:
            padded = arr
        padded_arrays.append(padded)
    stacked = np.stack(padded_arrays)
    return to_tensor_with_correct_dtype(stacked)


class NetworkXJSONDatabase(Database):
    """
    :param directory: directory path where the files are stored
    :param prefix: prefix for the files.

    This function loads files of the format f"{prefix}*.json" in directory.

    Other arguments: See ``Database``.
    """

    def __init__(self, directory, prefix, inputs, targets, *args, quiet=False, allow_unfound=False, **kwargs):

        self.quiet = quiet
        self.allow_unfound = allow_unfound

        arr_dict = self.load_files(directory, prefix, inputs, targets)
        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)


    def get_file_list(self, directory, prefix):
        try:
            file_list = os.listdir(directory)
        except FileNotFoundError as fee:
            raise FileNotFoundError(
                "ERROR: Couldn't find directory {} containing files."
                'A solution is to explicitly specify "path" in database_params '.format(directory)
            ) from fee

        data_files = {os.path.join(directory, file) for file in file_list if file.startswith(prefix) and file.endswith(".json")}

        # Make sure we actually found some files
        if not data_files:
            raise FileNotFoundError(
                "No files found at {} .".format(directory) + "for database prefix {}".format(prefix)
            )
        return data_files
    
    def load_one(self, file):
        graph = load_json_graph(file, self.quiet)
        node_data, edge_data = nx_to_npz(graph)
        return node_data, edge_data
    
    def sort_by_key(self, node_data, edge_data, required_keys=None):
        def extract_keys(data_list, keys):
            extracted = {}
            for key in keys:
                values = []
                for i, d in enumerate(data_list):
                    if key not in d:
                        raise KeyError(f"Dictionary at index {i} is missing key: '{key}'")
                    values.append(d[key])
                extracted[key] = values
            return extracted

        if required_keys is None:
            node_keys = set().union(*(d.keys() for d in node_data))
            edge_keys = set().union(*(d.keys() for d in edge_data))
            all_keys = node_keys | edge_keys
        else:
            all_keys = set(required_keys)
        all_keys.update({"ids",})

        # Split keys into those found in node_data and edge_data
        node_keys = set().union(*(d.keys() for d in node_data))
        edge_keys = set().union(*(d.keys() for d in edge_data))

        required_node_keys = all_keys & node_keys
        required_edge_keys = all_keys & edge_keys
        missing_keys = all_keys - (node_keys | edge_keys)
        if missing_keys:
            raise KeyError(f"The following required keys are missing from both node and edge data: {missing_keys}")

        node_output = extract_keys(node_data, required_node_keys)
        edge_output = extract_keys(edge_data, required_edge_keys)

        return node_output, edge_output
    
    def load_files(self, directory, prefix, inputs, targets):

        var_list = inputs + targets
        # Make sure the path actually exists

        data_files = self.get_file_list(directory, prefix=prefix)

        graph_data = [self.load_one(file) for file in data_files]

        node_data, edge_data = zip(*graph_data)

        node_data, edge_data = self.sort_by_key(node_data, edge_data, required_keys=(None if self.allow_unfound else var_list))

        arr_dict = {}

        for key, value in node_data.items():
            try:
                arr_dict[key] = array_list_to_padded_tensor(value)
            except Exception as err:
                msg = f"Failed to convert data for {key} to a tensor: {err}"
                if self.allow_unfound:
                    print(msg)
                else:
                    raise ValueError(msg)

        max_n_atoms = arr_dict["ids"].max().item() + 1

        for key, value in edge_data.items():
            try:
                if key == "edges":
                    arr_dict['edges'] = pairs_values_to_sparse(value, max_n_atoms=max_n_atoms)
                else:
                    value = to_tensor_with_correct_dtype([val for sublist in value for val in sublist])
                    arr_dict[key] = pairs_values_to_sparse(edge_data["edges"], value, max_n_atoms=max_n_atoms)
            except Exception as err:
                msg = f"Failed to convert data for {key} to a tensor: {err}"
                if self.allow_unfound:
                    print(msg)
                else:
                    raise ValueError(msg)
            
        if not self.quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})

        return arr_dict