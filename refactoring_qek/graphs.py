"""
Loading graphs as raw data.
"""

from __future__ import annotations
import abc
import logging
import math
from typing import Any, Final, Generic, TypeVar
import networkx as nx
import numpy as np
import pulser as pl
import rdkit.Chem as Chem
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from rdkit.Chem import AllChem
from qek.target import targets
from qek.shared._utils import graph_to_mol


logger = logging.getLogger(__name__)

EPSILON_RADIUS_UM = 0.01 # Assumption of rounding error when determining whether a graph is a disk graph.
EPSILON_RESCALE_FACTOR = 1.000000001 # A correction factor, attempting to cover for rounding error when rescaling a graph.
NODE_MAPPING: Final[dict[int, str]] = {
    0: "C",
    1: "S",
}

EDGE_MAPPING: dict[int, Chem.BondType] = {
    0: Chem.BondType.TRIPLE,
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.AROMATIC,
}

class BaseGraph:
    """
    A graph being prepared for embedding on a quantum device.
    """

    device: Final[pl.devices.Device]

    def __init__(
        self, id: int, data: pyg_data.Data, device: pl.devices.Device, target: float | None = None
    ):
        """
        Create a graph from geometric data.

        Args:
            id: An identifier for this graph, used mostly for error messages.
            data:  A homogeneous graph, in PyTorch Geometric format. Unchanged.
                It MUST have attributes 'pos'.
            device: The device for which the graph is prepared.
        """
        if not hasattr(data, "pos"):
            raise AttributeError("The graph should have an attribute 'pos'.")

        # The device for which the graph is prepared.
        self.device = device

        # The graph in torch geometric format.
        self.pyg = data.clone()

        # The graph in networkx format, undirected.
        self.nx_graph: nx.Graph = pyg_utils.to_networkx(
            data=data,
            node_attrs=["x"] if data.x is not None else None,
            edge_attrs=["edge_attr"] if data.edge_attr is not None else None,
            to_undirected=True,
        )
        self.target = target
        self.id = id

    def is_unit_disk_graph(self) -> bool:
        """
        A predicate to check if `self` is a unit disk graph.

        Returns:
            `True` if the graph is a unit disk graph.
            `False` otherwise.
        """

        if self.pyg.num_nodes == 0 or self.pyg.num_nodes is None:
            logger.debug("graph %s doesn't have any nodes, it's not a disk graph", self.id, self.id)
            return False

        # Check if the graph is connected.
        if len(self.nx_graph) == 0 or not nx.is_connected(self.nx_graph):
            logger.debug("graph %s is not connected, it's not a disk graph", self.id)
            return False

        # Check the distances between all pairs of nodes.
        pos = self.pyg.pos
        assert pos is not None

        non_connected_distances_um = [
            np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            for u, v in nx.non_edges(self.nx_graph)
        ]

        # Fully connected graphs are always unit disk graphs
        if len(non_connected_distances_um) == 0:
            return True

        connected_distances_um = [
            np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) for u, v in self.nx_graph.edges()
        ]

        if min(non_connected_distances_um) < max(connected_distances_um):
            return False

        return True

    def is_disk_graph(self, radius: float) -> bool:
        """
        A predicate to check if `self` is a disk graph with the specified
        radius, i.e. `self` is a connected graph and, for every pair of nodes
        `A` and `B` within `graph`, there exists an edge between `A` and `B`
        if and only if the positions of `A` and `B` within `self` are such
        that `|AB| <= radius`.

        Args:
            radius: The maximal distance between two nodes of `self`
                connected be an edge.

        Returns:
            `True` if the graph is a disk graph with the specified radius,
            `False` otherwise.
        """

        if self.pyg.num_nodes == 0 or self.pyg.num_nodes is None:
            logger.debug("graph %s doesn't have any nodes, it's not a disk graph", self.id, self.id)
            return False

        # Check if the graph is connected.
        if len(self.nx_graph) == 0 or not nx.is_connected(self.nx_graph):
            logger.debug("graph %s is not connected, it's not a disk graph", self.id)
            return False

        # Check the distances between all pairs of nodes.
        pos = self.pyg.pos
        assert pos is not None
        for u, v in nx.non_edges(self.nx_graph):
            distance_um = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance_um <= radius:
                # These disjointed nodes would interact with each other, so
                # this is not an embeddable graph.
                logger.debug(
                    "graph %s has non-edges that are too close to each other, it's not a disk graph",
                    self.id,
                )
                return False

        for u, v in self.nx_graph.edges():
            distance_um = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance_um > radius:
                # These joined nodes would not interact with each other, so
                # this is not an embeddable graph.
                logger.debug(
                    "graph %s has edges that are too distant from each other (%s > %s), it's not a disk graph",
                    self.id,
                    distance_um,
                    radius,
                )
                return False

        return True

    def is_embeddable(self) -> bool:
        """
            A predicate to check if the graph can be embedded in the
            quantum device.

            For a graph to be embeddable on a device, all the following
            criteria must be fulfilled:
            - the graph must be non-empty;
            - the device must have at least as many atoms as the graph has
                nodes;
            - the device must be physically large enough to place all the
                nodes (device.max_radial_distance);
            - the nodes must be distant enough that quantum interactions
                may take place (device.min_atom_distance)

        Returns:
            bool: True if possible, False if not
        """

        # Reject empty graphs.
        if self.pyg.num_nodes == 0 or self.pyg.num_nodes is None:
            logger.debug("graph %s is empty, it's not embeddable", self.id)
            return False

        # Reject graphs that have more nodes than can be represented
        # on the device.
        if self.pyg.num_nodes > self.device.max_atom_num:
            logger.debug(
                "graph %s has too many nodes (%s), it's not embeddable", self.id, self.pyg.num_nodes
            )
            return False

        # Check the distance from the center
        pos = self.pyg.pos
        assert pos is not None
        distance_from_center = np.linalg.norm(pos, ord=2, axis=-1)
        if any(distance_from_center > self.device.max_radial_distance):
            logger.debug(
                "graph %s has nodes to far from the center (%s > %s), it's not embeddable",
                self.id,
                max(distance_from_center),
                self.device.max_radial_distance,
            )
            return False

        # Check distance between nodes
        if not self.is_unit_disk_graph():
            logger.debug(
                "graph %s is not a unit disk graph, therefore it's not embeddable", self.id
            )
            return False

        for u, v in self.nx_graph.edges():
            distance_um = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance_um < self.device.min_atom_distance:
                # These nodes are too close to each other, preventing quantum
                # interactions on the device.
                logger.debug(
                    "graph %s has nodes that are too close to each other (%s < %s), it's not embeddable",
                    self.id,
                    distance_um,
                    self.device.min_atom_distance,
                )
                return False

        return True

    # Default values for the sequence.
    #
    # See the companion paper for an explanation.
    SEQUENCE_DEFAULT_AMPLITUDE_RAD_PER_US = 1.0 * 2 * np.pi
    SEQUENCE_DEFAULT_DURATION_NS = 660

    def compile_register(self) -> targets.Register:
        """Create a Quantum Register based on a graph.

        Returns:
            pulser.Register: register
        """
        # Note: In the low-level API, we separate register and pulse compilation for
        # pedagogical reasons, because we want to take the opportunity to teach them
        # about registers and pulses, rather than pulser sequences.

        if not self.is_embeddable():
            raise Exception(f"The graph is not compatible with {self.device}")

        # Compile register
        pos = self.pyg.pos
        assert pos is not None
        reg = pl.Register.from_coordinates(coords=pos)
        if self.device.requires_layout:
            reg = reg.with_automatic_layout(device=self.device)

        try:
            # Due to issue #29, we can produce a register that will not work on this device,
            # so we need to perform a second check.
            pl.Sequence(register=reg, device=self.device)
        except ValueError as e:
            raise Exception(f"The graph is not compatible with {self.device}: {e}")
        return targets.Register(device=self.device, register=reg)

    def compile_pulse(
        self,
        normalized_amplitude: float | None = None,
        normalized_duration: float | None = None,
    ) -> targets.Pulse:
        """Extract a Pulse for this graph.

        A Pulse represents the laser applied to the atoms on the device.

        Arguments:
            normalized_amplitude (optional): The normalized amplitude for the laser pulse, as a value in [0, 1],
                where 0 is no pulse and 1 is the maximal amplitude for the device. By default,
                use the value demonstrated in the companion paper.
            normalized_duration (optional): The normalized duration of the laser pulse, as a value in [0, 1],
                where 0 is the shortest possible duration and 1 is the longest possible
                duration. By default, use the value demonstrated in the companion paper.
        """
        # Note: In the low-level API, we separate register and pulse compilation for
        # pedagogical reasons, because we want to take the opportunity to teach them
        # about registers and pulses, rather than pulser sequences.

        channel = self.device.channels["rydberg_global"]
        assert channel is not None

        max_amp = channel.max_amp
        assert max_amp is not None

        min_duration = channel.min_duration
        max_duration = channel.max_duration
        assert max_duration is not None

        if normalized_amplitude is None:
            absolute_amplitude = self.SEQUENCE_DEFAULT_AMPLITUDE_RAD_PER_US
            if absolute_amplitude > max_amp:
                # Unlikely, but let's defend in depth.
                raise ValueError(
                    f"This device does not support pulses with amplitude {absolute_amplitude} rad per us, amplitudes should be <= {max_amp}"
                )
        else:
            if normalized_amplitude < 0 or normalized_amplitude > 1:
                raise ValueError("Invalid amplitude, expected a value in [0, 1] or None")
            absolute_amplitude = normalized_amplitude * max_amp

        if normalized_duration is None:
            absolute_duration = self.SEQUENCE_DEFAULT_DURATION_NS
            if absolute_duration < min_duration or absolute_duration > max_duration:
                # Unlikely, but let's defend in depth.
                raise ValueError(
                    f"This device does not support pulses with duration {absolute_duration} ns, pulses should be within [{min_duration}, {max_duration}]"
                )
        else:
            if normalized_duration < 0 or normalized_duration > 1:
                raise ValueError("Invalid duration, expected a value in [0, 1] or None")
            absolute_duration = (
                math.ceil(normalized_duration * (max_duration - min_duration)) + min_duration
            )

        # For an explanation on these constants, see the companion paper.
        pulse = pl.Pulse.ConstantAmplitude(
            amplitude=absolute_amplitude,
            detuning=pl.waveforms.RampWaveform(absolute_duration, 0, 0),
            phase=0.0,
        )
        return targets.Pulse(pulse)

class MoleculeGraph(BaseGraph):
    def __init__(self, id: Any, data: pyg_data.Data, device: pl.devices.Device, target: float | None = None):
        pyg = data.clone()
        pyg.pos = None
        super().__init__(id=id, data=pyg, device=device, target=target)

        tmp_mol = graph_to_mol(graph=self.nx_graph, node_mapping=NODE_MAPPING, edge_mapping=EDGE_MAPPING)

        AllChem.Compute2DCoords(tmp_mol, useRingTemplates=True)
        pos = tmp_mol.GetConformer().GetPositions()[..., :2]

        pairs = list(self.nx_graph.edges()) + list(nx.non_edges(self.nx_graph))
        min_distance = min(np.linalg.norm(pos[s] - pos[e]) for s, e in pairs)
        pos = pos * device.min_atom_distance / min_distance

        while True:
            min_distance = min(np.linalg.norm(pos[s] - pos[e]) for s, e in pairs)
            if min_distance >= device.min_atom_distance:
                break
            pos = pos * EPSILON_RESCALE_FACTOR

        self.pyg.pos = pos
        self.target = target

GraphType = TypeVar("GraphType")

class BaseGraphCompiler(abc.ABC, Generic[GraphType]):
    """
    Abstract class, used to load a graph and compile a Pulser sequence for a device.

    You should probably use one of the subclasses.
    """

    @abc.abstractmethod
    def ingest(self, graph: GraphType, device: pl.devices.Device, id: int) -> BaseGraph:
        raise Exception("Please use one of the subclasses")

class MoleculeGraphCompiler(BaseGraphCompiler[tuple[pyg_data.Data, float | None]]):
    def ingest(self, graph: tuple[pyg_data.Data, float | None] | pyg_data.Data, device: pl.devices.Device, id: int) -> MoleculeGraph:
        if isinstance(graph, tuple):
            data, target = graph
        else:
            data, target = graph, None
            if hasattr(data, "y") and data.y is not None:
                target = float(data.y.item() if isinstance(data.y, torch.Tensor) else data.y)

        return MoleculeGraph(id=id, data=data, device=device, target=target)
