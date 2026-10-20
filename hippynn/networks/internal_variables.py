"""
HIP-HOP-NN for particles carrying internal variables, such as orientations.
"""
from .hiphop import HipHopnnModule
from ..layers.hiplayers import HOPInteractionLayerOrient


class HipHopnnOrientModule(HipHopnnModule):
    """
    HIP-HOP-NN for particles with orientations (e.g. ellipsoids).

    Each particle has an orientation R_i whose columns are its body axes in the lab frame.
    The interaction layers use the relative orientation Q_ij = R_i^T R_j of each pair,
    so the model is invariant under a simultaneous global rotation of positions and orientations.

    .. Note::
       Each body axis is treated as having a direction. Physical ellipsoids are also unchanged by
       principal-axis sign flips (R_i -> R_i D, D diagonal with entries +-1);
       this linear expansion in Q_ij does not enforce that equivalence.
    """

    _interaction_class = HOPInteractionLayerOrient

    def forward(self, features, pair_first, pair_second, pair_dist, pair_coord, orientations):
        """
        :param orientations: (n_atoms, 3, 3) per-particle orientations R_i
        """
        # Q_ij = R_i^T R_j; (Q_ij)_ab is the dot product of axis a of i with axis b of j.
        orientations = orientations.to(pair_dist.dtype)
        pair_orient = (orientations[pair_first].transpose(1, 2) @ orientations[pair_second]).reshape(-1, 9)
        return super().forward(features, pair_first, pair_second, pair_dist, pair_coord, pair_orient)
