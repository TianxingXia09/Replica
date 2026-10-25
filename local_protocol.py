from __future__ import annotations

from itertools import combinations, product
import numpy as np


PAULI_EIGENVECTORS = {
    0: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # X
    1: np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2),  # Y
    2: np.eye(2, dtype=complex),  # Z
}


def all_subsystems(n: int, max_size: int | None = None) -> list[tuple[int, ...]]:
    max_size = n if max_size is None else max_size
    return [s for k in range(1, max_size + 1) for s in combinations(range(n), k)]


def random_density(d: int, rng: np.random.Generator, rank: int | None = None) -> np.ndarray:
    rank = d if rank is None else rank
    x = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    rho = x @ x.conj().T
    return rho / np.trace(rho)


def partial_trace(rho: np.ndarray, keep: tuple[int, ...], n: int) -> np.ndarray:
    keep = tuple(sorted(keep))
    traced = tuple(q for q in range(n) if q not in keep)
    tensor = rho.reshape((2,) * (2 * n))
    current_n = n
    for q in sorted(traced, reverse=True):
        tensor = np.trace(tensor, axis1=q, axis2=q + current_n)
        current_n -= 1
    return tensor.reshape(2 ** len(keep), 2 ** len(keep))


def subsystem_purities(rho: np.ndarray, subsets: list[tuple[int, ...]], n: int) -> np.ndarray:
    return np.array([
        float(np.real(np.trace((red := partial_trace(rho, s, n)) @ red)))
        for s in subsets
    ])


def basis_unitary(setting: tuple[int, ...]) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for axis in setting:
        # Columns are eigenvectors; U maps eigenbasis amplitudes to computational amplitudes.
        out = np.kron(out, PAULI_EIGENVECTORS[axis])
    return out


def probability_table(rho: np.ndarray, n: int) -> np.ndarray:
    table = np.empty((3,) * n + (2**n,), dtype=float)
    for setting in product(range(3), repeat=n):
        u = basis_unitary(setting)
        p = np.real(np.diag(u.conj().T @ rho @ u))
        table[setting] = np.clip(p, 0, None) / p.sum()
    return table


def sample_pair(
    table_left: np.ndarray,
    table_right: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    _, left, right = sample_pair_with_setting(table_left, table_right, n, rng)
    return left, right


def sample_pair_with_setting(
    table_left: np.ndarray,
    table_right: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a shared product-Pauli setting and both replica outcomes."""
    setting = tuple(rng.integers(0, 3, size=n))
    p = table_left[setting]
    q = table_right[setting]
    a = rng.choice(2**n, p=p)
    b = rng.choice(2**n, p=q)
    return (
        np.asarray(setting, dtype=np.int8),
        np.array([(a >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8),
        np.array([(b >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8),
    )


def sample_single(
    table: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one product-Pauli setting and its computational outcome bits."""
    setting = np.asarray(rng.integers(0, 3, size=n), dtype=np.int8)
    p = table[tuple(int(x) for x in setting)]
    outcome = int(rng.choice(2**n, p=p))
    bits = np.asarray(
        [(outcome >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8
    )
    return setting, bits


def local_overlap_kernel(
    a: np.ndarray, b: np.ndarray, subsets: list[tuple[int, ...]]
) -> np.ndarray:
    local = np.where(a == b, 2.0, -1.0)
    return np.array([np.prod(local[list(s)]) for s in subsets], dtype=float)


def corrected_local_overlap_kernel(
    a: np.ndarray,
    b: np.ndarray,
    subsets: list[tuple[int, ...]],
    error_rates: np.ndarray,
) -> np.ndarray:
    error_rates = np.asarray(error_rates, dtype=float)
    if np.any((error_rates < 0) | (error_rates >= 0.5)):
        raise ValueError("Each symmetric readout error rate must lie in [0, 1/2).")
    observed = np.where(a == b, 2.0, -1.0)
    attenuation = (1 - 2 * error_rates) ** 2
    corrected = 0.5 + (observed - 0.5) / attenuation
    return np.array([np.prod(corrected[list(s)]) for s in subsets], dtype=float)

