from __future__ import annotations

from itertools import product
import numpy as np

from local_protocol import PAULI_EIGENVECTORS


LOCAL_SHADOWS = np.empty((3, 2, 2, 2), dtype=complex)
for _axis in range(3):
    for _bit in range(2):
        _vec = PAULI_EIGENVECTORS[_axis][:, _bit]
        _projector = np.outer(_vec, _vec.conj())
        LOCAL_SHADOWS[_axis, _bit] = 3.0 * _projector - np.eye(2)


def _kron_local_shadow(
    settings_row: np.ndarray,
    bits_row: np.ndarray,
    subset: tuple[int, ...],
) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for qubit in subset:
        out = np.kron(
            out,
            LOCAL_SHADOWS[int(settings_row[qubit]), int(bits_row[qubit])],
        )
    return out


def shadow_pair_kernel(
    settings_left: np.ndarray,
    bits_left: np.ndarray,
    settings_right: np.ndarray,
    bits_right: np.ndarray,
    subset: tuple[int, ...],
) -> float:
    """Trace inner product of two local-Pauli classical shadows."""
    value = 1.0
    for qubit in subset:
        if int(settings_left[qubit]) != int(settings_right[qubit]):
            value *= 0.5
        elif int(bits_left[qubit]) == int(bits_right[qubit]):
            value *= 5.0
        else:
            value *= -4.0
    return float(value)


def windowed_shadow_u_statistics(
    settings: np.ndarray,
    bits: np.ndarray,
    subsets: list[tuple[int, ...]],
    windows: list[int],
) -> dict[int, np.ndarray]:
    """Complete U-statistics inside prespecified, non-overlapping windows.

    A block statistic averages Tr(shadow_i shadow_j) over i != j. It is
    unbiased for the corresponding average of cross-time state overlaps and
    for purity only when the states in the block are identical.
    """
    total_rounds = int(settings.shape[0])
    requested = sorted(set(int(w) for w in windows))
    if any(w < 2 for w in requested):
        raise ValueError("Every U-statistic window must contain at least two shots.")
    outputs = {w: np.empty(len(subsets), dtype=float) for w in requested}

    for subset_index, subset in enumerate(subsets):
        dimension = 2 ** len(subset)
        features = np.empty((total_rounds, dimension * dimension), dtype=complex)
        for row in range(total_rounds):
            features[row] = _kron_local_shadow(
                settings[row], bits[row], subset
            ).reshape(-1)
        cumulative = np.vstack(
            [np.zeros((1, features.shape[1]), dtype=complex), np.cumsum(features, axis=0)]
        )
        self_second_moment = float(5 ** len(subset))

        for width in requested:
            weighted = 0.0
            used = 0
            for start in range(0, total_rounds, width):
                stop = min(start + width, total_rounds)
                block_size = stop - start
                if block_size < 2:
                    continue
                summed = cumulative[stop] - cumulative[start]
                trace_square = float(np.vdot(summed, summed).real)
                estimate = (
                    trace_square - block_size * self_second_moment
                ) / (block_size * (block_size - 1))
                weighted += block_size * estimate
                used += block_size
            outputs[width][subset_index] = weighted / used
    return outputs


def bell_swap_probabilities(
    purity_by_subset: dict[tuple[int, ...], float], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact joint law of local SWAP eigenvalues on rho tensor rho.

    The commuting local SWAP signs have moments
    E[prod_{i in A} s_i] = Tr(rho_A^2). The inverse Walsh transform gives
    their full probability mass function.
    """
    signs = np.asarray(list(product((-1, 1), repeat=n)), dtype=np.int8)
    probabilities = np.zeros(len(signs), dtype=float)
    all_subsets = list(purity_by_subset)
    for row, sign_vector in enumerate(signs):
        total = 0.0
        for subset in all_subsets:
            character = 1 if not subset else int(np.prod(sign_vector[list(subset)]))
            total += float(purity_by_subset[subset]) * character
        probabilities[row] = total / (2**n)
    if probabilities.min() < -1e-9:
        raise ValueError("Invalid local-SWAP moment sequence.")
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    return signs, probabilities


def sample_bell_swap_signs(
    signs: np.ndarray,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return signs[int(rng.choice(len(signs), p=probabilities))]


def bell_swap_subset_kernel(
    local_signs: np.ndarray, subsets: list[tuple[int, ...]]
) -> np.ndarray:
    return np.asarray(
        [np.prod(local_signs[list(subset)]) for subset in subsets], dtype=float
    )


def global_rm_collision_from_purity(
    purity: float, subsystem_size: int, rng: np.random.Generator
) -> float:
    """Exact marginal simulation of the Haar/2-design purity collision.

    For d=2^k, the collision probability averaged over a unitary 2-design is
    (1+purity)/(d+1), and the estimator is d on collision and -1 otherwise.
    """
    dimension = 2**subsystem_size
    collision_probability = (1.0 + float(purity)) / (dimension + 1.0)
    return float(dimension if rng.random() < collision_probability else -1.0)
