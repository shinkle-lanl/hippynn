"""
Nodes for inputting information to the graph.
"""
import warnings
from .base import InputNode
from ..indextypes import IdxType
from .tags import Charges, Positions, Species, PairCache


class SpeciesNode(Species, InputNode):
    index_state = IdxType.SysAtom
    input_type_str = "Species"


class PositionsNode(Positions, InputNode):
    index_state = IdxType.SysAtom
    input_type_str = "Positions"


class OrientationsNode(InputNode):
    """Per-particle orientations, shape (n_systems, n_atoms, 3, 3); columns are body axes in the lab frame."""

    index_state = IdxType.SysAtom
    input_type_str = "Orientations"


class CellNode(InputNode):
    index_state = IdxType.Systems
    input_type_str = "Cells"

class ForceNode(InputNode):
    index_state = IdxType.SysAtom
    input_type_str = "Force"


class InputCharges(Charges, InputNode):
    index_state = IdxType.SysAtom
    input_type_str = "InputCharges"


class SystemIndices(InputNode):
    index_state = IdxType.Systems
    input_type_str = "Index"

    def __init__(self):
        super().__init__(db_name="indices")


class PairIndices(PairCache, InputNode):
    index_state = IdxType.Unlabeled
    input_type_str = "PairIndices"


class SplitIndices(InputNode):
    index_state = IdxType.Systems
    input_type_str = "Index"

    def __init__(self):
        super().__init__(db_name="indices")
