"""
Training example for HIP-HOP-NN with per-particle orientations (e.g. ellipsoids).

Each particle has an orientation (frame) R_i: a 3x3 rotation matrix whose columns are its body axes.
Passing the orientations to HIP-HOP-NN lets its interactions depend on the relative orientation
Q_ij = R_i^T R_j of each neighboring pair.

This script trains three models on the same synthetic data and compares them on the test set:

    1. HIP-HOP                 positions only (cannot see the orientations)
    2. HIP-HOP + orientations  positions and orientations, trained on energies
    3. + torque training       as model 2, trained on energies and torques

All three are evaluated on energies and forces; the two orientation models also on torques.

The data are clusters of 8 particles with a made-up pair energy that depends on distance
and on how the two particles' axes line up:

    E = sum over pairs of  phi(r_ij) * (1 + Q_ij[0,0] + 0.5 * Q_ij[1,2])
    phi(r) = exp(-r) * cos^2(pi r / 2 r_cutoff)

Runs in about fifteen minutes on a CPU.

NOTE: Each body axis is treated as having a direction. A physical ellipsoid looks the same if one of its axes is
reversed (R_i -> R_i D, D = diag(+-1, +-1, +-1)), but Q_ij changes sign, so the model does not
enforce that equivalence (and neither does the toy energy). If it matters for your data, pick a
consistent sign for each axis, or add copies of the data with flipped axes.
"""
import contextlib

import numpy as np
import torch

import hippynn
from hippynn.graphs import inputs, networks, internal_variables, targets, physics, loss
from hippynn.experiment import SetupParams, setup_and_train

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

N_SYSTEMS = 1000
N_PARTICLES = 8
CUTOFF = 3.0
OUTPUT_DIR = "TEST_ORIENTATION_HIPHOP"


# ----- Synthetic data -----


def random_rotations(*shape):
    """Uniformly random 3x3 rotation matrices."""
    q, r = torch.linalg.qr(torch.randn(*shape, 3, 3, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)  # makes q uniform over orthogonal matrices
    return q * torch.linalg.det(q)[..., None, None]  # flip reflections (det = -1) into rotations


def toy_energy(positions, orientations):
    """Total energy of each system: the pair energy above, summed over all pairs."""
    r = torch.cdist(positions, positions)
    q = torch.einsum("sixa,sjxb->sijab", orientations, orientations)  # q[s, i, j] = R_i^T R_j
    phi = torch.exp(-r) * torch.cos(np.pi * r / (2 * CUTOFF)) ** 2 * (r < CUTOFF)
    pair_energy = phi * (1 + q[..., 0, 0] + 0.5 * q[..., 1, 2])
    pair_energy = pair_energy * (1 - torch.eye(N_PARTICLES, dtype=orientations.dtype))  # no self-interaction
    return 0.5 * pair_energy.sum(dim=(1, 2))  # each pair appears twice in the sum


def make_data(n_systems):
    """Particles on a slightly jittered 2x2x2 cube, each with a random orientation."""
    cube = torch.stack(torch.meshgrid(*[torch.arange(2.0, dtype=torch.float64)] * 3, indexing="ij"), dim=-1)
    cube = cube.reshape(1, N_PARTICLES, 3)
    positions = (1.4 * cube + 0.1 * torch.randn(n_systems, N_PARTICLES, 3, dtype=torch.float64)).requires_grad_()
    orientations = random_rotations(n_systems, N_PARTICLES).requires_grad_()
    energy = toy_energy(positions, orientations)

    # Forces, and torques computed the same way physics.TorqueNode does for the model:
    # tau_i = -sum_k u_k x dE/du_k, where u_k are the body axes (columns of R_i).
    position_grad, orientation_grad = torch.autograd.grad(energy.sum(), (positions, orientations))
    torque = -torch.linalg.cross(orientations, orientation_grad, dim=-2).sum(dim=-1)

    return {
        "Z": torch.ones(n_systems, N_PARTICLES, dtype=torch.int64),  # species: one particle type
        "R": positions.detach().float(),
        "A": orientations.detach().float(),
        "E": energy.detach().float().unsqueeze(1),
        "F": -position_grad.float(),
        "T": torque.detach().float(),
    }


data = make_data(N_SYSTEMS)


# ----- Model and training -----

# The same settings for all three models.
network_params = {
    "possible_species": [0, 1],
    "n_features": 20,
    "n_sensitivities": 10,
    "dist_soft_min": 0.7,
    "dist_soft_max": 2.6,
    "dist_hard_max": CUTOFF,
    "n_interaction_layers": 1,
    "n_atom_layers": 2,
    "l_max": 2,
    "n_max": 3,
}


def train(name, use_orientations, train_on_torques=False):
    """Build and train one model, and return its test-set metrics."""
    torch.manual_seed(0)  # same initial weights and data split for every model

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    if use_orientations:
        # HipHopnnOrient takes the orientations as an extra input after the usual HIP-HOP-NN inputs.
        orientations = inputs.OrientationsNode(db_name="A")
        network = internal_variables.HipHopnnOrient(name, (species, positions, orientations), module_kwargs=network_params)
    else:
        network = networks.HipHopnn(name, (species, positions), module_kwargs=network_params)
    energy = targets.HEnergyNode("HEnergy", network, db_name="E")

    if use_orientations:
        # Forces and the orientation gradient from a single backward pass.
        gradients = physics.MultiGradientNode("gradients", energy.system_energy, (positions, orientations), signs=(-1, 1))
        force, orientation_grad = gradients.children
        torque = physics.TorqueNode("torque", (orientations, orientation_grad), db_name="T")
    else:
        force = physics.GradientNode("force", (energy, positions), sign=-1)
    force.db_name = "F"

    energy_mse = loss.MSELoss.of_node(energy)
    metrics = {
        "RMSE": energy_mse ** (1 / 2),
        "MAE": loss.MAELoss.of_node(energy),
        "Force-RMSE": loss.MSELoss.of_node(force) ** (1 / 2),
        "Force-MAE": loss.MAELoss.of_node(force),
    }
    training_loss = energy_mse

    if use_orientations:
        torque_mse = loss.MSELoss.of_node(torque)
        metrics["Torque-RMSE"] = torque_mse ** (1 / 2)
        metrics["Torque-MAE"] = loss.MAELoss.of_node(torque)
        if train_on_torques:
            training_loss = energy_mse + torque_mse  # equal weights; tune for real data

    training_modules, db_info = hippynn.experiment.assemble_for_training(training_loss, metrics)

    database = hippynn.databases.Database(
        arr_dict=data,
        seed=0,
        test_size=0.1,
        valid_size=0.1,
        pin_memory=False,
        quiet=True,
        **db_info,
    )

    print(f"Training {name}; progress is written to {OUTPUT_DIR}/{name}/training_log.txt")
    with hippynn.tools.active_directory(f"{OUTPUT_DIR}/{name}"):
        # Write the per-epoch training output to the log file only, to keep the console readable.
        with open("training_log.txt", "wt") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            metric_tracker = setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=SetupParams(
                    stopping_key="RMSE",  # every model keeps the epoch with the best energy RMSE
                    batch_size=32,
                    optimizer=torch.optim.Adam,
                    max_epochs=150,
                    learning_rate=1e-3,
                ),
            )

    return metric_tracker.other_metric_values["FinalTraining"]["test"]


results = {
    "HIP-HOP": train("hiphop", use_orientations=False),
    "HIP-HOP + orientations": train("hiphop_orient", use_orientations=True),
    "+ torque training": train("hiphop_orient_torque", use_orientations=True, train_on_torques=True),
}


# ----- Results ----- (-: the positions-only model has no orientations, so no torques)
keys = ("RMSE", "MAE", "Force-RMSE", "Force-MAE", "Torque-RMSE", "Torque-MAE")
print(f"\n{'Test set':<24}{'E RMSE':>9}{'E MAE':>9}{'F RMSE':>9}{'F MAE':>9}{'T RMSE':>9}{'T MAE':>9}")
for name, r in results.items():
    print(f"{name:<24}" + "".join(f"{r[k]:>9.4f}" if k in r else f"{'-':>9}" for k in keys))
