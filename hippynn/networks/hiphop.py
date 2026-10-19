import warnings
import torch
from .hipnn import Hipnn
from ..layers.hiplayers import HOPInteractionLayer, HOPInteractionLayerOrient, TensorExtractor


class HipHopnnModule(Hipnn):
    _interaction_class = HOPInteractionLayer
    _interaction_kwargs = ("l_max", "n_max", "group_norm", "group_norm_eps")

    def __init__(self, *args, l_max=3, n_max=4, group_norm=True, group_norm_eps=1e-5, **kwargs):
        warnings.warn("HIP-HOP-NN is still in a beta state: " "Details, defaults, and API are still subject to change.")
        super().__init__(*args, l_max=l_max, n_max=n_max, group_norm=group_norm, group_norm_eps=group_norm_eps, **kwargs)
        self.l_max = l_max
        self.n_max = n_max
        self.tensor_extractor = TensorExtractor(l_max=l_max)

    def extra_repr(self):
        return f"n_max={self.n_max}, l_max={self.l_max}"

    def forward(self, features, pair_first, pair_second, pair_dist, pair_coord, *pair_extras):
        """
        :param pair_extras: additional pair quantities passed through to each interaction layer
        """
        features = features.to(pair_dist.dtype)  # Convert one-hot features to floating point features.

        if pair_dist.ndim == 2:
            pair_dist = pair_dist.squeeze(dim=1)

        if pair_coord.ndim == 3:
            pair_coord = pair_coord.squeeze(dim=2)

        output_features = [features]
        rhats = pair_coord / pair_dist.unsqueeze(1)

        tensor_term_list = self.tensor_extractor(rhats)
        tensor_term_list = tensor_term_list[: self.l_max + 1]

        tensor_rhats = torch.cat(tensor_term_list, dim=-1)

        for block in self.blocks:
            int_layer = block[0]
            atom_layers = block[1:]

            features = int_layer(features, pair_first, pair_second, pair_dist, tensor_rhats, *pair_extras)
            if not self.resnet:
                features = self.activation(features)
            for lay in atom_layers:
                features = lay(features)
                if not self.resnet:
                    features = self.activation(features)
            output_features.append(features)

        return output_features


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


def __getattr__(name: str):
    import warnings
    if name == "HipHopNNModule":
        # Backwards compatibility for unpickling prior models
        warnings.warn(
            "'HipHopNNModule' is a deprecated class name, please use 'HipHopnnModule'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return HipHopnnModule
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
