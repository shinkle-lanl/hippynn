"""
Nodes for particles carrying internal variables, such as orientations.
"""
from .networks import HipHopnn
from .. import IdxType
from ... import networks as network_modules


class HipHopnnOrient(HipHopnn):
    """
    Node for HIP-HOP-NN for particles with orientations, using the relative orientation of each pair.

    Parents are those accepted by :class:`~hippynn.graphs.nodes.networks.HipHopnn`, followed by an
    :class:`~hippynn.graphs.nodes.inputs.OrientationsNode`, e.g. ``(species, positions, orientations)``.
    """

    input_names = "input_features", "pair_first", "pair_second", "pair_dist", "pair_coord", "orientations"
    auto_module_class = network_modules.internal_variables.HipHopnnOrientModule
    trailing_idx_states = (IdxType.Atoms,)
