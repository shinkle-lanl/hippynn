"""
Interaction layers for particles carrying internal variables, such as orientations.
"""
import torch
from ... import custom_kernels
from .interactions import HOPInteractionLayer


class HOPInteractionLayerOrient(HOPInteractionLayer):
    """
    HIP-HOP interaction layer for particles with orientations R_i.

    Each tensor sensitivity s_nu(r_ij) T_k(rhat_ij) is modulated by the relative orientation
    Q_ij = R_i^T R_j, so the message weights become::

        V0_{nu,ab} + sum_{alpha,beta} V^(alpha,beta)_{nu,ab} Q_ij[alpha,beta]

    ``int_weights`` holds V0, so this layer reduces exactly to :class:`HOPInteractionLayer`
    when ``orient_weights`` is zero. ``orient_weights[3 * alpha + beta]`` holds V^(alpha,beta).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orient_weights = torch.nn.Parameter(torch.Tensor(9, self.n_dist, self.nf_out, self.nf_in))
        # Initialize each orientation channel like the scalar channel.
        for w in self.orient_weights.data:
            torch.nn.init.xavier_normal_(w)

    def regularization_params(self):
        return [*super().regularization_params(), self.orient_weights]

    def forward(self, in_features, pair_first, pair_second, dist_pairs, tensor_rhats, pair_orient):
        """
        :param pair_orient: (n_pairs, 9) flattened relative orientations Q_ij
        """
        features_out_selfpart = self.selfint(in_features)

        n_atoms_real = in_features.shape[0]
        n_pair, n_tensor_comp = tensor_rhats.shape

        # Channels [1, Q_ab] for each tensor sensitivity, shape (n_pairs, n_tensor * 10 * n_nu)
        orient_channels = torch.cat([torch.ones_like(pair_orient[:, :1]), pair_orient], dim=1)
        sense_scalar = self.sensitivity(dist_pairs)
        sensitivity = tensor_rhats[:, :, None, None] * orient_channels[:, None, :, None] * sense_scalar[:, None, None, :]
        sense_flat = sensitivity.reshape(n_pair, n_tensor_comp * 10 * self.n_dist)

        env_features = custom_kernels.envsum(sense_flat, in_features, pair_first, pair_second)

        # apply weights to tensor features, in the same channel order as the sensitivities
        weights = torch.cat([self.int_weights.unsqueeze(0), self.orient_weights], dim=0)
        weights_rs = weights.permute(0, 1, 3, 2).reshape(10 * self.n_dist * self.nf_in, self.nf_out)
        env_rs = env_features.reshape(n_atoms_real * n_tensor_comp, 10 * self.n_dist * self.nf_in)
        tensor_features = torch.mm(env_rs, weights_rs)
        tensor_features = tensor_features.reshape(n_atoms_real, n_tensor_comp, self.nf_out)

        return self._mix_invariants(tensor_features) + features_out_selfpart
