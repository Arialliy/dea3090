#!/usr/bin/env python3
"""Audit lossless fixed-coordinate partitions for TRACE Stage 0D.

The audited object is deliberately narrower than a set-valued TRACE model.
Let every pixel coordinate be assigned to one fixed (not necessarily spatially
connected) cell.  For every observed mask we require (i) all pixels of each
8-connected component to have the same cell and (ii) distinct components to
have different cells.  The number of cells is unrestricted.

The equality constraints in (i) generate a union-find quotient.  Such a fixed
partition exists if and only if no inequality from (ii) becomes a self-loop in
that quotient.  This tool produces a replayable path certificate for the first
self-loop.  It also audits the *different* spatial-adjacency graph of quotient
classes: passing the union-find test would not make independently activated
cells a valid distribution over maximal 8-connected components.

Only official-training masks and actual one-epoch internal-fit augmentation
streams are read.  No official-test manifest, image, or mask is resolved or
opened anywhere in this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image
from skimage import measure
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import IRSTD_Dataset


DEFAULT_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
DEFAULT_SEEDS = (20260711, 20260712, 20260713)
DEFAULT_SPLIT_SEED = 20260711
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SIZE = 256
DEFAULT_BATCH_SIZE = 4


@dataclass(frozen=True)
class MaskObservation:
    """One labelled mask represented by row-major component supports."""

    height: int
    width: int
    components: tuple[tuple[int, ...], ...]
    metadata: dict
    binary_sha256: str


@dataclass(frozen=True)
class ComponentOccurrence:
    occurrence_id: int
    observation_index: int
    component_index: int
    pixels: tuple[int, ...]
    metadata: dict


@dataclass(frozen=True)
class InequalityConstraint:
    constraint_id: int
    occurrence_a: int
    occurrence_b: int
    coordinate_a: int
    coordinate_b: int
    observation_index: int


class UnionFind:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("union-find size must be positive")
        self.parent = list(range(size))
        self.size = [1] * size
        self.class_count = size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, first: int, second: int) -> bool:
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return False
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]
        self.class_count -= 1
        return True


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ordered_ids_sha256(names: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(names) + "\n").encode("utf-8"))


def parse_csv(text: str, cast) -> list:
    values = [cast(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def read_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"empty training manifest: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate IDs in training manifest: {path}")
    return names


def resolve_official_train_file(dataset_dir: Path) -> Path:
    """Resolve only the repository's official-training manifest candidates."""

    dataset_name = dataset_dir.name
    candidates = [
        dataset_dir / "trainval.txt",
        dataset_dir / "img_idx" / f"train_{dataset_name}.txt",
    ]
    candidates.extend(
        Path(item)
        for item in sorted(
            glob.glob(str(dataset_dir / "img_idx" / "train_*.txt"))
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "official-training manifest not found; tried: "
        + ", ".join(str(item) for item in candidates)
    )


def deterministic_internal_split(
    source_names: Sequence[str], split_seed: int, val_fraction: float
) -> tuple[list[str], list[str]]:
    if len(source_names) < 2:
        raise ValueError("at least two official-training samples are required")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie strictly between zero and one")
    ranked = sorted(
        source_names,
        key=lambda name: hashlib.sha256(
            f"{split_seed}\0{name}".encode("utf-8")
        ).digest(),
    )
    count = max(
        1,
        min(
            len(source_names) - 1,
            int(round(len(source_names) * val_fraction)),
        ),
    )
    validation = set(ranked[:count])
    return (
        [name for name in source_names if name not in validation],
        [name for name in source_names if name in validation],
    )


def binary_from_pil(mask: Image.Image, size: int) -> np.ndarray:
    resized = mask.resize((size, size), Image.NEAREST)
    tensor = transforms.ToTensor()(resized)
    return (tensor[0].numpy() > 0.5).astype(np.uint8)


def observation_from_binary(binary: np.ndarray, metadata: dict) -> MaskObservation:
    binary = np.asarray(binary, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    binary = (binary > 0).astype(np.uint8, copy=False)
    labels = measure.label(binary, connectivity=2, background=0)
    components = tuple(
        tuple(int(item) for item in np.flatnonzero(labels.reshape(-1) == label))
        for label in range(1, int(labels.max()) + 1)
    )
    if any(not component for component in components):
        raise AssertionError("component extraction emitted an empty support")
    return MaskObservation(
        height=int(binary.shape[0]),
        width=int(binary.shape[1]),
        components=components,
        metadata=dict(metadata),
        binary_sha256=sha256_bytes(binary.tobytes(order="C")),
    )


def load_official_train_observations(
    dataset_dir: Path, train_file: Path, size: int
) -> tuple[list[MaskObservation], list[str]]:
    names = read_names(train_file)
    observations: list[MaskObservation] = []
    for index, name in enumerate(names):
        mask_path = dataset_dir / "masks" / f"{name}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with Image.open(mask_path) as source:
            binary = binary_from_pil(source, size)
        observations.append(
            observation_from_binary(
                binary,
                {
                    "source": "official_train_eval_resize",
                    "sample_id": name,
                    "source_order_index": index,
                    "resize": [size, size],
                    "resize_mode": "nearest",
                },
            )
        )
    return observations, names


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _NamedTrainingMasks(Dataset):
    """Return IDs with the unmodified repository training transformation."""

    def __init__(self, base: IRSTD_Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        # Calling the base dataset (including image blur) is essential: it
        # consumes exactly the Python RNG sequence used by real training.
        _, mask = self.base[index]
        return mask, self.base.names[index], index


def make_train_args(
    dataset_dir: Path,
    train_file: Path,
    seed: int,
    split_seed: int,
    val_fraction: float,
    size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=str(dataset_dir.resolve()),
        evaluation_protocol="internal_holdout",
        train_split_file=str(train_file.resolve()),
        val_split_file="",
        val_fraction=val_fraction,
        split_seed=split_seed,
        seed=seed,
        base_size=size,
        crop_size=size,
        return_instance_map=False,
    )


def load_augmented_epoch_observations(
    dataset_dir: Path,
    train_file: Path,
    seed: int,
    split_seed: int,
    val_fraction: float,
    size: int,
    batch_size: int,
) -> tuple[list[MaskObservation], dict]:
    """Replay one actual single-process, shuffled internal-fit epoch."""

    _seed_everything(seed)
    args = make_train_args(
        dataset_dir, train_file, seed, split_seed, val_fraction, size
    )
    base = IRSTD_Dataset(args, mode="train")
    named = _NamedTrainingMasks(base)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        named,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    observations: list[MaskObservation] = []
    stream_index = 0
    for batch_index, (masks, names, source_indices) in enumerate(loader):
        for in_batch_index, (mask, name, source_index) in enumerate(
            zip(masks, names, source_indices.tolist())
        ):
            binary = (mask[0].numpy() > 0.5).astype(np.uint8)
            observations.append(
                observation_from_binary(
                    binary,
                    {
                        "source": "actual_augmented_internal_fit_epoch",
                        "seed": int(seed),
                        "epoch_zero_based": 0,
                        "sample_id": str(name),
                        "fit_source_index": int(source_index),
                        "stream_index": stream_index,
                        "batch_index": batch_index,
                        "in_batch_index": in_batch_index,
                    },
                )
            )
            stream_index += 1
    return observations, {
        "seed": int(seed),
        "fit_count_before_drop_last": len(base),
        "observed_count": len(observations),
        "dropped_by_drop_last": len(base) - len(observations),
        "fit_ordered_ids_sha256": base.split_sha256,
    }


def coordinate(pixel: int, width: int) -> list[int]:
    return [int(pixel // width), int(pixel % width)]


def occurrence_record(occurrence: ComponentOccurrence, width: int) -> dict:
    pixels = occurrence.pixels
    rows = [pixel // width for pixel in pixels]
    columns = [pixel % width for pixel in pixels]
    support_bytes = np.asarray(pixels, dtype="<u4").tobytes(order="C")
    return {
        "occurrence_id": occurrence.occurrence_id,
        "observation_index": occurrence.observation_index,
        "component_index_zero_based": occurrence.component_index,
        "component_label_one_based": occurrence.component_index + 1,
        "area": len(pixels),
        "anchor_coordinate": coordinate(pixels[0], width),
        "bounding_box_inclusive": [
            int(min(rows)),
            int(min(columns)),
            int(max(rows)),
            int(max(columns)),
        ],
        "support_flat_indices_uint32_le_sha256": sha256_bytes(support_bytes),
        "metadata": occurrence.metadata,
    }


def equality_path(
    start: int,
    goal: int,
    coordinate_to_occurrences: Sequence[Sequence[int]],
    occurrences: Sequence[ComponentOccurrence],
    width: int,
) -> dict:
    """Find a coordinate--component-hyperedge path proving equality."""

    queue: deque[int] = deque([start])
    parent: dict[int, tuple[int, int] | None] = {start: None}
    expanded_occurrences: set[int] = set()
    while queue and goal not in parent:
        current = queue.popleft()
        for occurrence_id in coordinate_to_occurrences[current]:
            if occurrence_id in expanded_occurrences:
                continue
            expanded_occurrences.add(occurrence_id)
            occurrence = occurrences[occurrence_id]
            for neighbor in occurrence.pixels:
                if neighbor not in parent:
                    parent[neighbor] = (current, occurrence_id)
                    queue.append(neighbor)
    if goal not in parent:
        raise AssertionError("union-find equality has no hypergraph path")

    reverse: list[tuple[int, int, int]] = []
    cursor = goal
    while cursor != start:
        previous, occurrence_id = parent[cursor]  # type: ignore[misc]
        reverse.append((previous, occurrence_id, cursor))
        cursor = previous
    transitions = list(reversed(reverse))
    path_occurrences: list[int] = []
    for _, occurrence_id, _ in transitions:
        if not path_occurrences or path_occurrences[-1] != occurrence_id:
            path_occurrences.append(occurrence_id)
    distinct_observations = {
        occurrences[item].observation_index for item in path_occurrences
    }
    return {
        "start_coordinate": coordinate(start, width),
        "end_coordinate": coordinate(goal, width),
        "transitions": [
            {
                "from_coordinate": coordinate(previous, width),
                "through_component_occurrence_id": occurrence_id,
                "to_coordinate": coordinate(current, width),
            }
            for previous, occurrence_id, current in transitions
        ],
        "component_chain": [
            occurrence_record(occurrences[item], width)
            for item in path_occurrences
        ],
        "component_chain_length": len(path_occurrences),
        "distinct_observation_count": len(distinct_observations),
        "distinct_observation_indices": sorted(distinct_observations),
        "spans_multiple_equality_observations": len(distinct_observations) > 1,
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(item) for item in values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def union_class_statistics(
    union_find: UnionFind, touched: set[int]
) -> tuple[dict, list[int], set[int]]:
    roots = [union_find.find(item) for item in range(len(union_find.parent))]
    sizes = Counter(roots)
    touched_roots = {roots[item] for item in touched}
    touched_sizes = [sizes[root] for root in touched_roots]
    return (
        {
            "coordinate_count": len(roots),
            "equality_class_count": len(sizes),
            "touched_coordinate_count": len(touched),
            "untouched_singleton_class_count": len(roots) - len(touched),
            "classes_touching_observed_foreground": len(touched_roots),
            "nontrivial_class_count": sum(value > 1 for value in sizes.values()),
            "largest_class_size": max(sizes.values(), default=0),
            "largest_touched_class_size": max(touched_sizes, default=0),
            "class_size_histogram": _histogram(sizes.values()),
            "touched_class_size_histogram": _histogram(touched_sizes),
        },
        roots,
        touched_roots,
    )


def spatial_adjacency_statistics(
    roots: Sequence[int],
    touched_roots: set[int],
    height: int,
    width: int,
    inequality_edges: set[tuple[int, int]],
    inequality_constraint_count: int,
) -> dict:
    """Build the 8-neighbour graph between distinct quotient classes."""

    edges: set[tuple[int, int]] = set()
    internal_pairs = 0
    grid_pairs = 0
    for row in range(height):
        for column in range(width):
            first = roots[row * width + column]
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                nr, nc = row + dr, column + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                grid_pairs += 1
                second = roots[nr * width + nc]
                if first == second:
                    internal_pairs += 1
                else:
                    edges.add((min(first, second), max(first, second)))

    degree: Counter[int] = Counter()
    for first, second in edges:
        degree[first] += 1
        degree[second] += 1
    touched_edges = {
        edge for edge in edges if edge[0] in touched_roots and edge[1] in touched_roots
    }
    boundary_edges = {
        edge for edge in edges if bool(edge[0] in touched_roots) != bool(edge[1] in touched_roots)
    }
    touched_degree: Counter[int] = Counter()
    for first, second in touched_edges:
        touched_degree[first] += 1
        touched_degree[second] += 1
    overlap = edges.intersection(inequality_edges)
    vertices = set(roots)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in vertices}
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    unvisited = set(vertices)
    connected_components = 0
    while unvisited:
        connected_components += 1
        start = unvisited.pop()
        queue = [start]
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
    return {
        "definition": (
            "Vertices are final equality quotient classes. An undirected edge "
            "joins two classes iff some coordinates, one in each class, are "
            "8-neighbours in the 256x256 lattice."
        ),
        "grid_8_neighbor_coordinate_pair_count": grid_pairs,
        "within_class_8_neighbor_coordinate_pair_count": internal_pairs,
        "quotient_vertex_count": len(vertices),
        "quotient_spatial_edge_count": len(edges),
        "quotient_spatial_connected_component_count": connected_components,
        "touched_induced_spatial_edge_count": len(touched_edges),
        "touched_to_untouched_spatial_edge_count": len(boundary_edges),
        "maximum_quotient_spatial_degree": max(degree.values(), default=0),
        "maximum_touched_induced_degree": max(touched_degree.values(), default=0),
        "unique_nonself_inequality_edge_count": len(inequality_edges),
        "spatial_and_inequality_edge_overlap_count": len(overlap),
        "merge_every_spatial_edge_would_collapse_the_grid_to_one_cell": (
            connected_components == 1
        ),
        "merge_every_spatial_edge_conflicts_with_observed_inequalities": (
            connected_components == 1 and inequality_constraint_count > 0
        ),
        "independent_product_warning": (
            "The union-find criterion only fits observed masks to fixed cells. "
            "If distinct cells may be simultaneously active with supports on "
            "an edge of this graph, those supports can merge into one maximal "
            "8-connected component. Independent Bernoulli cell activation is "
            "therefore not a valid exact set construction."
        ),
        "required_extension": (
            "Forbid simultaneous activation across every spatial edge and "
            "normalize a hard-core distribution globally, or introduce a "
            "provable inactive buffer / merge cells. The latter changes the "
            "support family. Because the quotient adjacency graph contracts a "
            "connected pixel lattice, merging every spatial edge produces one "
            "cell and violates any observed inter-component inequality. "
            "No tractability result for that global normalizer is established "
            "by this audit."
        ),
    }


def _make_conflict_certificate(
    inequality: InequalityConstraint,
    trigger: dict,
    occurrences: Sequence[ComponentOccurrence],
    coordinate_to_occurrences: Sequence[Sequence[int]],
    width: int,
) -> dict:
    first = occurrences[inequality.occurrence_a]
    second = occurrences[inequality.occurrence_b]
    path = equality_path(
        inequality.coordinate_a,
        inequality.coordinate_b,
        coordinate_to_occurrences,
        occurrences,
        width,
    )
    path["uses_cross_observation_equalities"] = any(
        item["observation_index"] != inequality.observation_index
        for item in path["component_chain"]
    )
    transition_membership_verified = all(
        previous in occurrences[occurrence_id].pixels
        and current in occurrences[occurrence_id].pixels
        for previous, occurrence_id, current in (
            (
                transition["from_coordinate"][0] * width
                + transition["from_coordinate"][1],
                transition["through_component_occurrence_id"],
                transition["to_coordinate"][0] * width
                + transition["to_coordinate"][1],
            )
            for transition in path["transitions"]
        )
    )
    return {
        "certificate_type": "inequality_self_loop_in_equality_quotient",
        "trigger": trigger,
        "inequality_constraint_id": inequality.constraint_id,
        "inequality_source_observation_index": inequality.observation_index,
        "component_a": occurrence_record(first, width),
        "component_b": occurrence_record(second, width),
        "equality_path": path,
        "verification_conditions": {
            "components_are_distinct_in_same_mask": (
                first.observation_index == second.observation_index
                and first.component_index != second.component_index
            ),
            "path_endpoints_match_inequality_anchors": (
                path["start_coordinate"]
                == coordinate(inequality.coordinate_a, width)
                and path["end_coordinate"]
                == coordinate(inequality.coordinate_b, width)
            ),
            "every_equality_transition_uses_one_observed_component": (
                transition_membership_verified
            ),
        },
    }


def audit_fixed_partition(observations: Sequence[MaskObservation]) -> dict:
    if not observations:
        raise ValueError("at least one mask observation is required")
    height, width = observations[0].height, observations[0].width
    if any((item.height, item.width) != (height, width) for item in observations):
        raise ValueError("all masks must share one coordinate domain")
    domain_size = height * width
    union_find = UnionFind(domain_size)
    occurrences: list[ComponentOccurrence] = []
    inequalities: list[InequalityConstraint] = []
    coordinate_to_occurrences: list[list[int]] = [
        [] for _ in range(domain_size)
    ]
    touched: set[int] = set()
    first_conflict: dict | None = None

    def find_loop() -> InequalityConstraint | None:
        for inequality in inequalities:
            if union_find.find(inequality.coordinate_a) == union_find.find(
                inequality.coordinate_b
            ):
                return inequality
        return None

    for observation_index, observation in enumerate(observations):
        current_occurrences: list[int] = []
        for component_index, pixels in enumerate(observation.components):
            occurrence_id = len(occurrences)
            occurrence = ComponentOccurrence(
                occurrence_id=occurrence_id,
                observation_index=observation_index,
                component_index=component_index,
                pixels=pixels,
                metadata=dict(observation.metadata),
            )
            occurrences.append(occurrence)
            current_occurrences.append(occurrence_id)
            anchor = pixels[0]
            for pixel in pixels:
                if not 0 <= pixel < domain_size:
                    raise ValueError("component coordinate lies outside domain")
                touched.add(pixel)
                coordinate_to_occurrences[pixel].append(occurrence_id)
                union_find.union(anchor, pixel)
            if first_conflict is None:
                loop = find_loop()
                if loop is not None:
                    first_conflict = _make_conflict_certificate(
                        loop,
                        {
                            "phase": "component_equality_hyperedge",
                            "trigger_observation_index": observation_index,
                            "trigger_component_index_zero_based": component_index,
                            "trigger_metadata": observation.metadata,
                        },
                        occurrences,
                        coordinate_to_occurrences,
                        width,
                    )

        for left in range(len(current_occurrences)):
            for right in range(left + 1, len(current_occurrences)):
                first_id = current_occurrences[left]
                second_id = current_occurrences[right]
                inequality = InequalityConstraint(
                    constraint_id=len(inequalities),
                    occurrence_a=first_id,
                    occurrence_b=second_id,
                    coordinate_a=occurrences[first_id].pixels[0],
                    coordinate_b=occurrences[second_id].pixels[0],
                    observation_index=observation_index,
                )
                inequalities.append(inequality)
                if (
                    first_conflict is None
                    and union_find.find(inequality.coordinate_a)
                    == union_find.find(inequality.coordinate_b)
                ):
                    first_conflict = _make_conflict_certificate(
                        inequality,
                        {
                            "phase": "component_inequality_edge",
                            "trigger_observation_index": observation_index,
                            "trigger_component_pair_zero_based": [left, right],
                            "trigger_metadata": observation.metadata,
                        },
                        occurrences,
                        coordinate_to_occurrences,
                        width,
                    )

    class_stats, roots, touched_roots = union_class_statistics(
        union_find, touched
    )
    self_loops = [
        item
        for item in inequalities
        if roots[item.coordinate_a] == roots[item.coordinate_b]
    ]
    nonself_edges = {
        (
            min(roots[item.coordinate_a], roots[item.coordinate_b]),
            max(roots[item.coordinate_a], roots[item.coordinate_b]),
        )
        for item in inequalities
        if roots[item.coordinate_a] != roots[item.coordinate_b]
    }
    component_histogram = Counter(len(item.components) for item in observations)
    feasible = not self_loops
    if feasible != (first_conflict is None):
        raise AssertionError(
            "incremental first-conflict detector disagrees with final quotient"
        )
    return {
        "observation_count": len(observations),
        "observation_component_count_histogram": {
            str(key): int(component_histogram[key])
            for key in sorted(component_histogram)
        },
        "component_occurrence_count": len(occurrences),
        "inequality_constraint_count_with_multiplicity": len(inequalities),
        "final_self_loop_constraint_count_with_multiplicity": len(self_loops),
        "arbitrary_fixed_partition_feasible": feasible,
        "decision_reason": (
            "no inequality self-loop in the equality quotient"
            if feasible
            else "at least one inequality is a self-loop in the equality quotient"
        ),
        "first_conflict_certificate": first_conflict,
        "union_classes": class_stats,
        "quotient_spatial_8_adjacency": spatial_adjacency_statistics(
            roots,
            touched_roots,
            height,
            width,
            nonself_edges,
            len(inequalities),
        ),
        "scope_guardrail": (
            "This decision concerns lossless slicing of the finite observed "
            "masks by arbitrary fixed coordinate cells only. It neither proves "
            "coverage of unobserved augmentations nor establishes an exact "
            "independent-cell distribution over maximal 8-connected sets."
        ),
    }


def restricted_growth_partitions(size: int) -> Iterator[tuple[int, ...]]:
    """Enumerate every labelled-normal-form set partition of ``range(size)``."""

    if size <= 0:
        return
    labels = [0] * size

    def visit(index: int, maximum: int) -> Iterator[tuple[int, ...]]:
        if index == size:
            yield tuple(labels)
            return
        for label in range(maximum + 2):
            labels[index] = label
            yield from visit(index + 1, max(maximum, label))

    yield from visit(1, 0)


def partition_satisfies(
    labels: Sequence[int], observations: Sequence[MaskObservation]
) -> bool:
    for observation in observations:
        component_cells: list[int] = []
        for component in observation.components:
            cells = {labels[pixel] for pixel in component}
            if len(cells) != 1:
                return False
            component_cells.append(next(iter(cells)))
        if len(component_cells) != len(set(component_cells)):
            return False
    return True


def brute_force_partition_exists(
    domain_size: int, observations: Sequence[MaskObservation]
) -> bool:
    return any(
        partition_satisfies(partition, observations)
        for partition in restricted_growth_partitions(domain_size)
    )


def exhaustive_small_grid_validation() -> dict:
    """Compare the theorem with every corpus of masks on a 1x3 grid."""

    masks = [
        observation_from_binary(
            np.array(
                [[(bits >> column) & 1 for column in range(3)]],
                dtype=np.uint8,
            ),
            {"mask_bits": bits},
        )
        for bits in range(8)
    ]
    mismatches: list[dict] = []
    for corpus_bits in range(1 << len(masks)):
        corpus = [
            masks[index]
            for index in range(len(masks))
            if (corpus_bits >> index) & 1
        ]
        # The empty corpus imposes no constraints and is trivially feasible;
        # audit_fixed_partition intentionally requires at least one observation.
        theorem = True if not corpus else bool(
            audit_fixed_partition(corpus)["arbitrary_fixed_partition_feasible"]
        )
        brute_force = brute_force_partition_exists(3, corpus)
        if theorem != brute_force:
            mismatches.append(
                {
                    "corpus_bits": corpus_bits,
                    "theorem": theorem,
                    "brute_force": brute_force,
                }
            )
    partition_count = sum(1 for _ in restricted_growth_partitions(3))
    return {
        "grid": [1, 3],
        "connectivity": 8,
        "possible_binary_masks": len(masks),
        "mask_corpora_exhaustively_checked": 1 << len(masks),
        "set_partitions_per_corpus": partition_count,
        "theorem_bruteforce_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def observation_stream_sha256(observations: Sequence[MaskObservation]) -> str:
    digest = hashlib.sha256()
    for observation in observations:
        digest.update(
            json.dumps(
                observation.metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(observation.binary_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_dataset(
    dataset_dir: Path,
    seeds: Sequence[int],
    *,
    split_seed: int,
    val_fraction: float,
    size: int,
    batch_size: int,
) -> dict:
    dataset_dir = dataset_dir.resolve()
    train_file = resolve_official_train_file(dataset_dir)
    official, official_names = load_official_train_observations(
        dataset_dir, train_file, size
    )
    expected_fit, expected_validation = deterministic_internal_split(
        official_names, split_seed, val_fraction
    )

    per_seed_observations: list[list[MaskObservation]] = []
    per_seed_provenance: list[dict] = []
    per_seed_audits: list[dict] = []
    for seed in seeds:
        observations, provenance = load_augmented_epoch_observations(
            dataset_dir,
            train_file,
            int(seed),
            split_seed,
            val_fraction,
            size,
            batch_size,
        )
        if provenance["fit_ordered_ids_sha256"] != ordered_ids_sha256(
            expected_fit
        ):
            raise AssertionError(
                "IRSTD_Dataset internal-fit split differs from independent reproduction"
            )
        provenance["observation_stream_sha256"] = observation_stream_sha256(
            observations
        )
        per_seed_observations.append(observations)
        per_seed_provenance.append(provenance)
        per_seed_audits.append(audit_fixed_partition(observations))

    augmented = [
        observation
        for observations in per_seed_observations
        for observation in observations
    ]
    combined = [*official, *augmented]
    return {
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "official_training_manifest": str(train_file),
        "official_training_manifest_sha256": sha256_bytes(train_file.read_bytes()),
        "official_training_ordered_ids_sha256": ordered_ids_sha256(
            official_names
        ),
        "internal_fit_ordered_ids_sha256": ordered_ids_sha256(expected_fit),
        "internal_validation_ordered_ids_sha256": ordered_ids_sha256(
            expected_validation
        ),
        "official_train_eval_resize": {
            "observation_stream_sha256": observation_stream_sha256(official),
            "audit": audit_fixed_partition(official),
        },
        "actual_augmented_internal_fit_one_epoch": {
            "loader_contract": {
                "implementation": "utils.data.IRSTD_Dataset",
                "epochs_per_seed": 1,
                "epoch_zero_based": 0,
                "seeds": [int(seed) for seed in seeds],
                "shuffle": True,
                "batch_size": batch_size,
                "drop_last": True,
                "num_workers": 0,
                "pin_memory": True,
                "generator_seed": "model seed",
                "base_size": size,
                "crop_size": size,
            },
            "per_seed_provenance": per_seed_provenance,
            "per_seed_audits": [
                {"seed": int(seed), "audit": audit}
                for seed, audit in zip(seeds, per_seed_audits)
            ],
            "aggregate_observation_stream_sha256": observation_stream_sha256(
                augmented
            ),
            "aggregate_audit": audit_fixed_partition(augmented),
        },
        "cumulative_official_train_plus_observed_augmentation": {
            "observation_stream_sha256": observation_stream_sha256(combined),
            "audit": audit_fixed_partition(combined),
        },
    }


def repository_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(
    dataset_dirs: Sequence[Path],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    size: int = DEFAULT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    exhaustive = exhaustive_small_grid_validation()
    if not exhaustive["passed"]:
        raise AssertionError("small-grid theorem/brute-force comparison failed")
    datasets = [
        audit_dataset(
            dataset_dir,
            seeds,
            split_seed=split_seed,
            val_fraction=val_fraction,
            size=size,
            batch_size=batch_size,
        )
        for dataset_dir in dataset_dirs
    ]
    official_all_infeasible = all(
        not item["official_train_eval_resize"]["audit"][
            "arbitrary_fixed_partition_feasible"
        ]
        for item in datasets
    )
    augmented_all_infeasible = all(
        not item["actual_augmented_internal_fit_one_epoch"][
            "aggregate_audit"
        ]["arbitrary_fixed_partition_feasible"]
        for item in datasets
    )
    every_seed_infeasible = all(
        not seed_item["audit"]["arbitrary_fixed_partition_feasible"]
        for item in datasets
        for seed_item in item["actual_augmented_internal_fit_one_epoch"][
            "per_seed_audits"
        ]
    )
    return {
        "schema_version": 1,
        "stage": "TRACE Stage 0D fixed-coordinate-partition feasibility",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "repository_head": repository_head(),
        "implementation_sources": {
            "repository_dataset_loader": str(
                (PROJECT_ROOT / "utils" / "data.py").resolve()
            ),
            "repository_dataset_loader_sha256": sha256_bytes(
                (PROJECT_ROOT / "utils" / "data.py").read_bytes()
            ),
        },
        "theorem": {
            "domain": (
                "An arbitrary fixed partition of the HxW coordinate set into "
                "an unrestricted number of cells. Cells need not be connected."
            ),
            "requirements": [
                "Every pixel of each observed 8-connected component is assigned to one cell.",
                "Distinct 8-connected components in the same observed mask are assigned to different cells.",
            ],
            "criterion": (
                "Union all coordinate pairs belonging to a common observed "
                "component. A satisfying fixed partition exists iff no "
                "same-mask inter-component inequality has both endpoints in "
                "one resulting union-find class."
            ),
            "necessity": (
                "Every satisfying cell assignment is constant on each primitive "
                "component equality and therefore on its transitive closure. "
                "An inequality self-loop would require one quotient class to "
                "receive both equal and unequal cell labels, a contradiction."
            ),
            "sufficiency": (
                "When there is no self-loop, assign a distinct cell label to "
                "every equality quotient class. All equalities hold by "
                "construction and every inequality connects two different "
                "classes, hence receives different labels."
            ),
            "unlimited_cell_assumption": True,
            "executable_exhaustive_check": exhaustive,
        },
        "protocol": {
            "mask_threshold": "ToTensor(mask)[0] > 0.5",
            "foreground_connectivity": 8,
            "official_training_resize": [size, size],
            "official_training_resize_mode": "nearest",
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "augmentation_seeds": [int(seed) for seed in seeds],
            "augmentation_epochs_per_seed": 1,
            "batch_size": batch_size,
            "drop_last": True,
            "num_workers": 0,
            "earliest_conflict_order": (
                "Process observations in manifest order (official train) or "
                "DataLoader order (augmentation). Within an observation, add "
                "each row-major 8CC equality hyperedge atomically in label "
                "order, checking all earlier inequalities after each; then add "
                "same-mask component-pair inequalities in lexicographic label "
                "order. The first infeasible prefix under this fixed order is "
                "reported."
            ),
        },
        "official_test_nonuse_guardrail": {
            "official_test_manifest_read": False,
            "official_test_images_read": False,
            "official_test_masks_read": False,
            "statement": (
                "The implementation exposes only an official-training resolver "
                "and constructs IRSTD_Dataset only in train mode. No test path "
                "is resolved, enumerated, hashed, or opened."
            ),
            "rng_equivalence_note": (
                "The normal trainer constructs validation and test Dataset "
                "objects before its loaders. Those constructors consume no RNG; "
                "they are intentionally omitted here to enforce test non-use, "
                "without changing the train augmentation RNG sequence."
            ),
        },
        "finite_stream_scope": (
            "The augmentation evidence contains exactly one deterministic epoch "
            "for each listed seed. A pass would certify only those finite "
            "observations, not all 400 epochs or the augmentation distribution's "
            "closure. A failure is a valid counterexample to the proposed fixed "
            "partition for the audited training protocol."
        ),
        "set_model_scope": (
            "The union-find theorem is not a construction of an exact set-valued "
            "TRACE product. Quotient-class spatial adjacency is audited separately; "
            "simultaneous activation across such edges requires a global hard-core "
            "normalizer or a different buffered/merged support family."
        ),
        "headline_decision": {
            "all_datasets_official_train_infeasible": official_all_infeasible,
            "all_datasets_augmented_aggregate_infeasible": augmented_all_infeasible,
            "every_dataset_seed_one_epoch_stream_infeasible": every_seed_infeasible,
            "augmentation_expansion_required": not every_seed_infeasible,
            "augmentation_expansion_decision": (
                "Not required: every individual dataset/seed one-epoch stream "
                "already contains a constructive counterexample. No claim about "
                "all 400 epochs is needed or made."
                if every_seed_infeasible
                else "A finite observed stream passed; expand deterministically before drawing a protocol conclusion."
            ),
        },
        "datasets": datasets,
    }


def resolve_dataset_dirs(tokens: Sequence[str], root: Path) -> list[Path]:
    output: list[Path] = []
    for token in tokens:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute() and not candidate.exists():
            candidate = root / token
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        output.append(candidate.resolve())
    return output


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether arbitrary fixed coordinate cells can losslessly "
            "separate all observed training components."
        )
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "datasets"
    )
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "repro_runs"
            / "clean"
            / "trace_stage0_fixed_partition_feasibility_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = resolve_dataset_dirs(
        parse_csv(args.datasets, str), args.dataset_root
    )
    seeds = parse_csv(args.seeds, int)
    report = build_report(datasets, seeds)
    output = args.output.expanduser().resolve()
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "datasets": [item["dataset"] for item in report["datasets"]],
                "small_grid_check_passed": report["theorem"][
                    "executable_exhaustive_check"
                ]["passed"],
                "official_test_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
