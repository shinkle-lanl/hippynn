Oriented Particles
==================

HIP-HOP-NN (:class:`~hippynn.graphs.nodes.networks.HipHopnn`) accepts an optional orientations input for
particles such as rigid ellipsoids. Each particle's orientation (its reference frame) R_i is a 3x3 rotation
matrix whose columns are the particle's body axes in the lab frame. It follows the spin extension of HIP-NN,
which modulates the interaction kernel with the bond variable S_i . S_j. Here, the full relative orientation
Q_ij = R_i^T R_j is used instead; its nine entries are the dot products between the axes of particle i and
the axes of particle j.

In HIP-HOP-NN, each pair contributes sensitivities s_nu(r_ij) T_k(rhat_ij), where T_k are the
irreducible Cartesian tensor components of the bond direction up to ``l_max``. Here each of these
is weighted by::

    V0_{nu,ab} + sum_{alpha,beta} V^(alpha,beta)_{nu,ab} Q_ij[alpha,beta]

That is, one scalar channel plus nine orientation channels, each with its own learned weights. The
resulting tensor features pass through the standard HIP-HOP invariants, group normalization and mixing.
Everything else (on-site layers, ResNet blocks, hierarchical energies, pair finding, and autograd forces)
is unchanged. If the orientation weights are zero, the network is exactly HIP-HOP-NN; with ``l_max=0``
only T_0 = 1 remains, and the kernel reduces to the HIP-NN form
v_ab(r_ij, Q_ij) = sum_nu s_nu(r_ij) [ V0_{nu,ab} + sum_{alpha,beta} V^(alpha,beta)_{nu,ab} Q_ij[alpha,beta] ].

Example::

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    orientations = inputs.OrientationsNode(db_name="A")  # shape (n_systems, n_atoms, 3, 3)

    network_params.update({"l_max": 2, "n_max": 3})
    network = networks.HipHopnn("hiphop_orient", (species, positions, orientations), module_kwargs=network_params)
    henergy = targets.HEnergyNode("HEnergy", network)

    # Forces and the orientation gradient from a single backward pass.
    gradients = physics.MultiGradientNode("gradients", henergy.system_energy, (positions, orientations), signs=(-1, 1))
    force, orientation_grad = gradients.children
    force.db_name = "F"
    torque = physics.TorqueNode("torque", (orientations, orientation_grad), db_name="T")

``TorqueNode`` computes tau_i = -sum_k u_k x dE/du_k from the energy gradient with respect to the
orientation (sign +1), where u_k are the body axes (columns of R_i). Only these three combinations of the
3x3 orientation gradient are physical; the other six would deform the body axes. It takes the gradient from a
``MultiGradientNode`` as above, so forces and torques share one backward pass, or from a ``GradientNode``
if only torques are needed. With a ``db_name``, torques can be added to the training loss in the same way
as forces, e.g. ``loss.MSELoss.of_node(torque)``.

Passing an :class:`~hippynn.graphs.nodes.inputs.OrientationsNode` as the last parent selects the
orientation-dependent module, :class:`~hippynn.networks.hiphop.HipHopnnOrientModule`; without it, the node
builds standard HIP-HOP-NN. The choice is made when the model is built, since the orientation weights are
part of the model.

``examples/orientation_hiphop_example.py`` trains HIP-HOP-NN without orientations, with orientations, and
with orientations plus torque training on the same synthetic data, and compares them on the test set.

Symmetry
--------

Rotating all positions and orientations together (r_i -> U r_i, R_i -> U R_i) leaves the model invariant.

Because orientations enter only through Q_ij, which is unchanged by R_i -> U R_i, the model is also
separately invariant to rotating all orientations while holding positions fixed. The bond tensors T_k(rhat_ij)
are formed in the lab frame and are never projected onto the body axes, so the model does not see the
direction of a bond relative to a particle's axes (for example, it cannot distinguish two parallel
ellipsoids that are side-by-side from two that are end-to-end at the same distance). This is the direct
analog of the spin extension, where spin and lattice rotations are decoupled.

Each body axis is treated as having a direction. A physical ellipsoid is unchanged when one of its principal
axes is reversed (R_i -> R_i D with D = diag(+-1, +-1, +-1)), but Q_ij changes sign in the corresponding rows or
columns, so the model does not enforce this equivalence. If it matters, canonicalize axis signs in the
data or augment the data with sign flips.
