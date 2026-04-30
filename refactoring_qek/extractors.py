import abc
from dataclasses import dataclass
import itertools
import json
import logging
from typing import Any, Callable, Generator, Generic, Sequence, TypeVar, cast
from numpy.typing import NDArray
from pathlib import Path
import numpy as np
import pulser as pl
from pulser.devices import Device
from pulser_simulation import QutipEmulator
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data

from qek.data import processed_data
from qek.data.processed_data import ProcessedData
from qek.shared.error import CompilationError
from qek_main.qek.data.graphs import NODE_MAPPING, EDGE_MAPPING, BaseGraph, BaseGraphCompiler
from rdkit import Chem
from rdkit.Chem import AllChem


logger = logging.getLogger(__name__)

@dataclass
class Compiled:
    """
    The result of compiling a graph for execution on a quantum device.
    """

    # Future plans: as of this writing, this class (or a reworked version of it)
    # is expected to move to the `qool-layer` library.

    # The graph itself.
    graph: BaseGraph

    # A sequence adapted to the quantum device.
    sequence: pl.Sequence


@dataclass
class Feature:
    """
    A feature extracted from raw data.
    """

    data: NDArray[np.floating]


class BaseExtracted(abc.ABC):
    """
    Data extracted by one of the subclasses of `BaseExtractor`.

    Note that the list of processed data will generally *not* contain all the graphs ingested
    by the Extractor, as not all graphs may not be compiled for a given device.
    """

    def __init__(self, device: Device):
        self.device = device

    def __await__(self) -> Generator[Any, Any, None]:
        """
        Wait asynchronously until execution is ready.

        This will avoid blocking your main thread, so calling this method once,
        before the first call to `processed_data`, is strongly recommended
        for use on a server or an interactive application.
        """
        # By default, no need to wait.
        yield None

    @property
    @abc.abstractmethod
    def processed_data(self) -> list[processed_data.ProcessedData]:
        pass

    @property
    @abc.abstractmethod
    def raw_data(self) -> list[BaseGraph]:
        """
        A subset of the graphs ingested by the Extractor.
        """
        pass

    @property
    @abc.abstractmethod
    def targets(self) -> list[int] | None:
        """
        If available, the machine-learning targets for these graphs, in the same order and with the same number of entrie as `raw_data`.
        """
        pass

    @property
    def states(self) -> list[dict[str, int]]:
        """
        The quantum states extracted from `raw_data` by executing `sequences` on the device, in the same order and with the same number of entries as `raw_data`.
        """
        return [data.state_dict for data in self.processed_data]

    def features(self, size_max: int | None) -> list[Feature]:
        """
        The features extracted from `raw_data` by processing `states`, in the same order and with the same number of entries as `raw_data`.

        By default, the features extracted are the distribution of excitation levels based on `states`. However, subclasses may override
        this method to provide custom features extraction.

        Arguments:
            size_max (optional) Performance/precision lever. If specified, specifies the number of qubits to take into account from all
                the `states`. If `size_max` is lower than the number of qubits used to extract `self.states[i]` (i.e. the number of qubits
                in `self.sequences[i]`), then only take into account the `size_max` first qubits of this state to extract
                `self.features(size_max)[i]`. If, on the other hand, `size_max` is greater than the number of qubits used to extract
                `self.states[i]`, pad `self.features(size_max)[i]` with 0s.
                If unspecified, use the largest number of qubits in `selfsequences`.
        """
        if size_max is None:
            for data in self.processed_data:
                seq = data._sequence
                if size_max is None or len(seq.qubit_info) > size_max:
                    size_max = len(seq.qubit_info)
        if size_max is None:
            # The only way size_max can be None is if `self.sequences` is empty.
            return []

        return [Feature(processed_data.dist_excitation(state, size_max)) for state in self.states]

    def save_dataset(self, file_path: Path) -> None:
        """Saves the processed dataset to a JSON file.

        Note: This does NOT attempt to save the graphs.

        Args:
            dataset: The dataset to be saved.
            file_path: The path where the dataset will be saved as a JSON
                file.

        Note:
            The data is stored in a format suitable for loading with load_dataset.
        """
        with open(file_path, "w") as file:
            states = self.states
            targets = self.targets
            data = [
                {
                    "sequence": self.processed_data[i]._sequence.to_abstract_repr(),
                    # Some emulators will actually be `dict[str, int64]` instead of `dict[str, int]` and `int64`
                    # is not JSON-serializable.
                    #
                    # The reason for which `int64` is not JSON-serializable is that JSON limits ints to 2^53-1.
                    # However, in practice, this should not be a problem, since the `int`/`int64` in our dict is
                    # limited to the number of runs, and we don't expect to be launching 2^53 consecutive runs
                    # for a single sequence on a device in any foreseeable future (assuming a run of 1ns,
                    # this would still take ~4 billion years to execute).
                    "state_dict": {key: int(value) for (key, value) in states[i].items()},
                    "target": targets[i] if targets is not None else None,
                }
                for i in range(len(self.processed_data))
            ]
            json.dump(data, file)
        logger.info("processed data saved to %s", file_path)


class SyncExtracted(BaseExtracted):
    """
    Data extracted synchronously, i.e. no need to wait for a remote server.
    """

    def __init__(
        self,
        raw_data: list[BaseGraph],
        targets: list[int] | None,
        sequences: list[pl.Sequence],
        states: list[dict[str, int]],
    ):
        assert len(raw_data) == len(sequences)
        assert len(sequences) == len(states)
        if targets is not None:
            if len(targets) < len(sequences):
                # Not all graphs come with a target.
                #
                # This Extracted will not be usable as the training sample, so ignore all targets.
                if len(targets) != 0:
                    logger.debug(
                        "We compiled %s graphs but we only have %s targets, ignoring all targets",
                        len(sequences),
                        len(targets),
                    )
                targets = None
        self._raw_data = raw_data
        self._targets = targets
        self._sequences = sequences
        self._states = states
        self._processed_data = [
            ProcessedData(
                sequence=seq, state_dict=cast(dict[str, int | np.int64], state), target=target
            )
            for (seq, state, target) in itertools.zip_longest(sequences, states, targets or [])
        ]

    @property
    def processed_data(self) -> list[ProcessedData]:
        return self._processed_data

    @property
    def raw_data(self) -> list[BaseGraph]:
        return self._raw_data

    @property
    def targets(self) -> list[int] | None:
        return self._targets

    @property
    def sequences(self) -> list[pl.Sequence]:
        return self._sequences

    @property
    def states(self) -> list[dict[str, int]]:
        return self._states


# Type variable for BaseExtractor[GraphType].
GraphType = TypeVar("GraphType")


class BaseExtractor(abc.ABC, Generic[GraphType]):
    """
    The base of the hierarchy of extractors.

    The role of extractors is to take a list of raw data (here, labelled graphs) into
    processed data containing machine-learning features (here, excitation vectors).

    Args:
        path: If specified, the processed data will be saved to this file as JSON once
            the execution is complete.
        device: A quantum device for which the data should be prepared.
        compiler: A graph compiler, in charge of converting graphs to Pulser Sequences,
            the format that can be executed on a quantum device.
    """

    def __init__(
        self, device: Device, compiler: BaseGraphCompiler[GraphType], path: Path | None = None
    ) -> None:
        self.path = path

        # The list of graphs (raw data). Fill it with `self.add_graphs`.
        self.graphs: list[BaseGraph] = []
        self.device: Device = device

        # The compiled sequences. Filled with `self.compile`.
        # Note that the list of compiled sequences may be shorter than the list of
        # raw data, as not all graphs may be compiled to a given `device`.
        self.sequences: list[Compiled] = []
        self.compiler = compiler

        # A counter used to give a unique id to each graph.
        self._counter = 0

    def save(self, snapshot: list[ProcessedData]) -> None:
        """Saves a dataset to a JSON file.

        Args:
            dataset (list[ProcessedData]): The dataset to be saved, containing
                RegisterData instances.
            file_path (str): The path where the dataset will be saved as a JSON
                file.

        Note:
            The data is stored in a format suitable for loading with load_dataset.
        """
        if self.path is not None:
            with open(self.path, "w") as file:
                data = [
                    {
                        "sequence": instance._sequence.to_abstract_repr(),
                        "state_dict": instance.state_dict,
                        "target": instance.target,
                    }
                    for instance in snapshot
                ]
                json.dump(data, file)
            logger.info("processed data saved to %s", self.path)

    def compile(
        self, filter: Callable[[BaseGraph, pl.Sequence, int], bool] | None = None
    ) -> list[Compiled]:
        """
        Compile all pending graphs into Pulser sequences that the Quantum Device may execute.

        Once this method has succeeded, the results are stored in `self.sequences`.
        """
        if len(self.graphs) == 0:
            raise Exception("No graphs to compile, did you forget to call `import_graphs`?")
        if filter is None:
            filter = lambda _graph, sequence, _index: True  # noqa: E731
        self.sequences = []
        for graph in self.graphs:
            try:
                register = graph.compile_register()
                pulse = graph.compile_pulse()
                sequence = pl.Sequence(register=register.register, device=graph.device)
                sequence.declare_channel("ising", "rydberg_global")
                sequence.add(pulse.pulse, "ising")
            except CompilationError as e:
                # In some cases, we produce graphs that pass `is_embeddable` but cannot be compiled.
                # It _looks_ like this is due to rounding errors. We're investigating this in issue #29,
                # but for the time being, we're simply logging and skipping them.
                logger.debug("Graph #%s could not be compiled (%s), skipping", graph.id, e)
                continue
            if not filter(graph, sequence, graph.id):
                logger.debug("Graph #%s did not pass filtering, skipping", graph.id)
                continue
            logger.debug("Compiling graph #%s for execution on the device", graph.id)
            self.sequences.append(Compiled(graph=graph, sequence=sequence))
        logger.debug("Compilation step complete, %s graphs compiled", len(self.sequences))
        return self.sequences

    # add_graphs
    def load_data(self, smiles_list: list[str], target_list: list[float]) -> None:
        """
        convert smiles to PyG type to compile and run.
        """
        # internal functions ======================================================================================
        def atom_to_one_hot(atom: Chem.Atom, node_mapping: dict[int, str]) -> np.ndarray:
            """Convert an RDKit atom into a one-hot vector based on NODE_MAPPING."""
            num_classes = len(node_mapping)
            one_hot = np.zeros(num_classes, dtype=np.float32)
            for idx, symbol in node_mapping.items():
                if atom.GetSymbol() == symbol:
                    one_hot[idx] = 1.0
                    break
            return one_hot

        def bond_to_one_hot(bond: Chem.Bond, edge_mapping: dict[int, Chem.BondType]) -> np.ndarray:
            """Convert an RDKit bond into a one-hot vector based on EDGE_MAPPING."""
            num_classes = len(edge_mapping)
            one_hot = np.zeros(num_classes, dtype=np.float32)
            for idx, btype in edge_mapping.items():
                if bond.GetBondType() == btype:
                    one_hot[idx] = 1.0
                    break
            return one_hot

        def smiles_to_pyg(smiles_list: list[str], target_list: list[float] | None = None) -> list[Data]:
            """
            Convert a list of SMILES strings and optional target values into
            a list of PyTorch Geometric Data objects with one-hot node and edge features.
            """
            data_list = []
            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES string at index {i}: {smiles}")

                AllChem.Compute2DCoords(mol)

                # Node features: one-hot encoding
                x = np.array([atom_to_one_hot(atom, NODE_MAPPING) for atom in mol.GetAtoms()], dtype=np.float32)

                # Edge list and attributes: one-hot encoding
                edge_index, edge_attr = [], []
                for bond in mol.GetBonds():
                    s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_index += [[s, e], [e, s]]  # undirected
                    edge_attr += [bond_to_one_hot(bond, EDGE_MAPPING)] * 2

                edge_index = np.array(edge_index, dtype=np.int64).T
                edge_attr = np.array(edge_attr, dtype=np.float32)

                # Target
                target = target_list[i] if target_list is not None else None
                y = np.array([target], dtype=np.float32) if target is not None else None

                # Positions (optional)
                # pos = np.array(mol.GetConformer().GetPositions()[..., :2], dtype=np.float32)

                # Wrap into PyG Data
                data = Data(
                    x=torch.from_numpy(x),
                    edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_attr),
                    y=torch.from_numpy(np.round(y,5)) if y is not None else None,
                    # pos=torch.from_numpy(pos),
                )
                data_list.append(data)

            return data_list
        # internal functions ======================================================================================
        
        graphs = smiles_to_pyg(smiles_list, target_list)
        for graph in graphs:
            self._counter += 1
            id = self._counter
            logger.debug("ingesting # %s", id)
            processed = self.compiler.ingest(graph=graph, device=self.device, id=id)
            # Skip graphs that are not embeddable.
            if processed.is_embeddable():
                logger.debug("graph # %s is embeddable, accepting", id)
                self.graphs.append(processed)
            else:
                logger.info("graph # %s is not embeddable, skipping", id)
        logger.info("imported %s graphs", len(self.graphs))

    @abc.abstractmethod
    def run(self) -> BaseExtracted:
        """
        Run compiled graphs.

        You will need to call `self.compile` first, to make sure that the graphs are compiled.

        Returns:
            Data extracted by this extractor.

            Not all extractors may return the same data, so please take a look at the documentation
            of the extractor you are using.
        """
        raise Exception("Not implemented")


class QutipExtractor(BaseExtractor[GraphType]):
    """
    A Extractor that uses the Qutip Emulator to run sequences compiled
    from graphs.
    
    Args:
        path: Path to store the result of the run, for future uses.
            To reload the result of a previous run, use `LoadExtractor`.
        compiler: A graph compiler, in charge of converting graphs to Pulser Sequences,
            the format that can be executed on a quantum device.
        device: A device to use. For general experiments, the default
            device `AnalogDevice` is a perfectly reasonable choice.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        device: Device = pl.devices.AnalogDevice,
        path: Path | None = None,
    ):
        super().__init__(path=path, device=device, compiler=compiler)
        self.graphs: list[BaseGraph]
        self.device = device

    def run(self, max_qubits: int = 8) -> SyncExtracted:
        """
        Run the compiled graphs.

        As emulating a quantum device is slow consumes resources and time exponential in the
        number of qubits, for the sake of performance, we limit the number of qubits in the execution
        of this extractor.

        Args:
            max_qubits: Skip any sequence that require strictly more than `max_qubits`. Defaults to 8.

        Returns:
            Processed data for all the sequences that were executed.
        """
        if len(self.sequences) == 0:
            logger.warning("No sequences to run, did you forget to call compile()?")
            return SyncExtracted(raw_data=[], targets=[], sequences=[], states=[])

        raw_data: list[BaseGraph] = []
        targets: list[float] = []
        sequences: list[pl.Sequence] = []
        states: list[dict[str, int]] = []
        for compiled in self.sequences:
            qubits_used = len(compiled.sequence.qubit_info)
            if qubits_used > max_qubits:
                logger.info(
                    "Graph %s exceeds the qubit limit specified in QutipExtractor (%s > %s), skipping",
                    id,
                    qubits_used,
                    max_qubits,
                )
                continue
            logger.debug("Executing compiled graph # %s", id)
            simul = QutipEmulator.from_sequence(sequence=compiled.sequence)
            counter = cast(dict[str, Any], simul.run().sample_final_state())
            logger.debug("Execution of compiled graph # %s complete", id)
            raw_data.append(compiled.graph)
            if compiled.graph.target is not None:
                targets.append(compiled.graph.target)
            sequences.append(compiled.sequence)
            states.append(counter)

        result = SyncExtracted(
            raw_data=raw_data, targets=targets, sequences=sequences, states=states
        )
        logger.debug("Emulation step complete, %s compiled graphs executed", len(raw_data))
        if self.path is not None:
            result.save_dataset(self.path)
        return result
