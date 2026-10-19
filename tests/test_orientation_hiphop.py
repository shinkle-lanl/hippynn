"""
Tests for HIP-HOP-NN with optional per-particle orientations (HipHopnnOrientModule).
"""
import pytest
import torch

from hippynn.networks.hiphop import HipHopnnModule, HipHopnnOrientModule

pytestmark = pytest.mark.filterwarnings("ignore:.*beta state.*")

NETWORK_PARAMS = {
    "possible_species": [0, 1, 2],
    "n_features": 6,
    "n_sensitivities": 5,
    "dist_soft_min": 0.75,
    "dist_soft_max": 2.0,
    "dist_hard_max": 2.5,
    "n_interaction_layers": 2,
    "n_atom_layers": 1,
    "l_max": 2,
    "n_max": 3,
}


def random_rotations(*shape):
    q, r = torch.linalg.qr(torch.randn(*shape, 3, 3, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)  # sign fix makes q uniform (Haar) on O(3)
    return q * torch.linalg.det(q)[..., None, None]  # det(-q) = -det(q) in 3D, so this maps O(3) uniformly onto SO(3)


def lattice_positions(*shape):
    # Jittered 2x2x2 lattice: no pairs below the sensitivity range, and the
    # body diagonals (~2.6) mostly fall outside the cutoff.
    grid = torch.stack(torch.meshgrid(*[torch.arange(2.0, dtype=torch.float64)] * 3, indexing="ij"), dim=-1)
    return 1.5 * grid.reshape(8, 3) + 0.05 * torch.randn(*shape, 8, 3, dtype=torch.float64)


def example_inputs():
    species = torch.randint(1, 3, (3, 8))
    species[1, -1] = 0  # padding
    return species, lattice_positions(3), random_rotations(3, 8)


def example_pairs(n_input_features):
    positions = lattice_positions()
    dist = torch.cdist(positions, positions)
    in_range = (dist < NETWORK_PARAMS["dist_hard_max"]) & ~torch.eye(len(positions), dtype=torch.bool)
    pair_first, pair_second = torch.nonzero(in_range, as_tuple=True)
    pair_coord = positions[pair_second] - positions[pair_first]
    features = torch.randn(len(positions), n_input_features, dtype=torch.float64)
    return features, pair_first, pair_second, dist[pair_first, pair_second], pair_coord


def make_energy_graph():
    from hippynn.graphs import inputs, networks, targets, GraphModule

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    orientations = inputs.OrientationsNode(db_name="A")
    network = networks.HipHopnn("HIPHOP_orient", (species, positions, orientations), module_kwargs=NETWORK_PARAMS)
    energy = targets.HEnergyNode("E", network)
    graph = GraphModule([species, positions, orientations], [energy.system_energy]).to(torch.float64)
    return graph, network.torch_module


def test_orientations_parent_selects_module():
    from hippynn.graphs import inputs, networks, IdxType

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    cell = inputs.CellNode(db_name="C")
    orientations = inputs.OrientationsNode(db_name="A")

    plain = networks.HipHopnn("plain", (species, positions), module_kwargs=NETWORK_PARAMS)
    assert type(plain.torch_module) is HipHopnnModule
    assert "orientations" not in plain.input_names

    for parents in [(species, positions, orientations), (species, positions, cell, orientations)]:
        net = networks.HipHopnn("orient", parents, module_kwargs=NETWORK_PARAMS, periodic=cell in parents)
        assert type(net.torch_module) is HipHopnnOrientModule
        assert net.orientations.index_state == IdxType.Atoms
        assert net.input_names == (*plain.input_names, "orientations")


def test_global_rotation_invariance():
    torch.manual_seed(0)
    graph, _ = make_energy_graph()
    species, positions, orientations = example_inputs()

    rotation = random_rotations()
    shift = torch.randn(3, dtype=torch.float64)
    energy = graph(species, positions, orientations)[0]
    energy_rotated = graph(species, positions @ rotation.T + shift, rotation @ orientations)[0]
    assert torch.allclose(energy, energy_rotated)

    # The energy depends on the orientations.
    assert not torch.allclose(energy, graph(species, positions, random_rotations(3, 8))[0])

    # Orientations enter only through Q_ij, so rotating the orientations alone is also a symmetry
    # (as with spins in mHIP-NN): the body axes are not coupled to the bond directions.
    assert torch.allclose(energy, graph(species, positions, rotation @ orientations)[0])


def reference_tensor_features(layer, tensor_extractor, features, pair_first, pair_second, pair_dist, pair_coord, orientations):
    """Direct evaluation of sum_j sum_b s_nu(r_ij) T_k(rhat_ij) [V0 + sum_ab V^(ab) Q_ij[a,b]] z_jb."""
    sense = layer.sensitivity(pair_dist)
    rhats = pair_coord / pair_dist.unsqueeze(1)
    tensors = torch.cat(tensor_extractor(rhats)[: tensor_extractor.l_max + 1], dim=-1)
    out = torch.zeros(len(features), tensors.shape[1], layer.nf_out, dtype=features.dtype)
    for p, (i, j) in enumerate(zip(pair_first.tolist(), pair_second.tolist())):
        kernel = torch.einsum("n,nxy->xy", sense[p], layer.int_weights)
        for a in range(3):
            for b in range(3):
                q_ab = orientations[i, :, a] @ orientations[j, :, b]  # axis a of particle i dotted with axis b of particle j
                kernel = kernel + q_ab * torch.einsum("n,nxy->xy", sense[p], layer.orient_weights[3 * a + b])
        out[i] += tensors[p].unsqueeze(1) * (kernel @ features[j]).unsqueeze(0)
    return out


def test_orientation_terms():
    torch.manual_seed(0)
    params = {**NETWORK_PARAMS, "n_interaction_layers": 1, "n_atom_layers": 0, "resnet": False}
    network = HipHopnnOrientModule(**params).to(torch.float64)
    layer = network.interaction_layers[0]
    features, pair_first, pair_second, pair_dist, pair_coord = example_pairs(network.nf_in)
    orientations = random_rotations(len(features))

    # Channel 0 is the scalar term V^(0); channel 1 + 3a + b is V^(ab).
    # Check each of the ten channels in isolation, then all at once.
    masks = [torch.arange(10) == channel for channel in range(10)] + [torch.ones(10, dtype=torch.bool)]
    for mask in masks:
        with torch.no_grad():
            weights = torch.randn(10, *layer.int_weights.shape, dtype=torch.float64)
            weights = weights * mask[:, None, None, None]
            layer.int_weights.copy_(weights[0])
            layer.orient_weights.copy_(weights[1:])

        out = network(features, pair_first, pair_second, pair_dist, pair_coord, orientations)[1]
        tensor_features = reference_tensor_features(
            layer, network.tensor_extractor, features, pair_first, pair_second, pair_dist, pair_coord, orientations
        )
        expected = network.activation(layer._mix_invariants(tensor_features) + layer.selfint(features))
        assert torch.allclose(out, expected)


def test_differentiable():
    torch.manual_seed(0)
    graph, network = make_energy_graph()
    species, positions, orientations = example_inputs()
    positions.requires_grad_(True)
    orientations.requires_grad_(True)

    def energy(positions, orientations):
        return graph(species, positions, orientations)[0]

    assert torch.autograd.gradcheck(energy, (positions, orientations), fast_mode=True)
    assert torch.autograd.gradgradcheck(energy, (positions, orientations), fast_mode=True)

    # The orientation coefficients receive gradients for training.
    energy(positions, orientations).sum().backward()
    for layer in network.interaction_layers:
        assert layer.orient_weights.grad.abs().sum() > 0


def test_torque():
    from hippynn.graphs import inputs, networks, targets, physics, GraphModule

    torch.manual_seed(0)
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    orientations = inputs.OrientationsNode(db_name="A")
    network = networks.HipHopnn("HIPHOP_orient", (species, positions, orientations), module_kwargs=NETWORK_PARAMS)
    energy = targets.HEnergyNode("E", network)
    # Forces and orientation gradients from a single backward pass.
    gradients = physics.MultiGradientNode("gradients", energy.system_energy, (positions, orientations), signs=(-1, 1))
    forces, orientation_grad = gradients.children
    torque = physics.TorqueNode("torque", (orientations, orientation_grad))
    reference_forces = physics.GradientNode("reference_forces", (energy, positions), sign=-1)
    graph = GraphModule(
        [species, positions, orientations], [energy.system_energy, torque, forces, reference_forces]
    ).to(torch.float64)
    # A force-style gradient (sign=-1) would give torques of the wrong sign.
    negated_grad = physics.GradientNode("negated_grad", (energy, orientations), sign=-1)
    with pytest.raises(ValueError):
        physics.TorqueNode("bad_torque", (orientations, negated_grad))

    z, r, orientations = example_inputs()
    _, torques, force_values, reference_force_values = graph(z, r, orientations)
    assert torch.allclose(force_values, reference_force_values)

    # The torque component about axis c is minus the derivative of the energy
    # when one particle's orientation is rotated about c by a small angle.
    eps = 1e-6
    sys_i, atom_i = 2, 3
    for c in range(3):
        axis = torch.eye(3, dtype=torch.float64)[c]
        generator = torch.linalg.cross(axis.expand(3, 3), torch.eye(3, dtype=torch.float64), dim=1).T  # v -> axis x v
        energies = []
        for sign in (1, -1):
            rotated = orientations.clone()
            rotated[sys_i, atom_i] = torch.linalg.matrix_exp(sign * eps * generator) @ orientations[sys_i, atom_i]
            energies.append(graph(z, r, rotated)[0][sys_i])
        finite_difference = -(energies[0] - energies[1]) / (2 * eps)
        assert torch.allclose(torques[sys_i, atom_i, c], finite_difference.squeeze(), atol=1e-7)

    # Rotating all orientations together is a symmetry, so the torques on each system sum to zero.
    assert torch.allclose(torques.sum(dim=1), torch.zeros(3, 3, dtype=torch.float64), atol=1e-12)


def test_zero_orientation_weights_recovers_hiphop():
    torch.manual_seed(0)
    plain = HipHopnnModule(**NETWORK_PARAMS).to(torch.float64)
    orient = HipHopnnOrientModule(**NETWORK_PARAMS).to(torch.float64)

    missing, unexpected = orient.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(k.endswith("orient_weights") for k in missing)
    with torch.no_grad():
        for layer in orient.interaction_layers:
            layer.orient_weights.zero_()

    features, pair_first, pair_second, pair_dist, pair_coord = example_pairs(plain.nf_in)
    orientations = random_rotations(len(features))
    plain_out = plain(features, pair_first, pair_second, pair_dist, pair_coord)
    orient_out = orient(features, pair_first, pair_second, pair_dist, pair_coord, orientations)
    for f_plain, f_orient in zip(plain_out, orient_out, strict=True):
        assert torch.allclose(f_plain, f_orient)
