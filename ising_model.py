"""
2D Ising Model simulation using JAX.

Hamiltonian: H = -J * sum_{<i,j>} s_i * s_j
  - J = 1 (ferromagnetic coupling, sets the energy scale)
  - s_i in {-1, +1} (spin-1/2 sites on a square lattice)
  - sum over nearest-neighbor pairs with periodic boundary conditions

Temperature is measured in units of J/k_B throughout.
The exact critical temperature (Onsager 1944) is:
    T_c = 2J / (k_B * ln(1 + sqrt(2))) ≈ 2.2692


"""

import time
import jax
import jax.numpy as jnp
from jax import jit, lax, random
import pandas as pd

jax.config.update("jax_enable_x64", True)


def run_ising_model(
    L: int = 36,
    MCS: int = 1000,
    T_start: float = 4.0,
    T_end: float = 0.5,
    dT: float = 0.05,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Simulate the 2D ferromagnetic Ising model using the Metropolis algorithm.

    The lattice is initialised in the fully ordered state (all spins +1) and
    then slowly cooled from T_start to T_end. Carrying the configuration over
    between temperatures reduces equilibration artefacts near the transition.

    One Monte Carlo sweep is implemented via checkerboard (sublattice) updates:
    all sites of sublattice A are updated simultaneously (via vectorised
    jnp.where), then all sites of sublattice B. Because nearest-neighbour sites
    always belong to opposite sublattices, updates within one sublattice are
    independent, making the simultaneous update statistically equivalent to
    random sequential updates.


    Parameters
    ----------
    L : int
        Linear lattice size; the lattice contains N = L x L spins.
    MCS : int
        Number of Monte Carlo sweeps per temperature point.
    T_start : float
        Starting temperature (in units of J/k_B), swept downward.
    T_end : float
        Final temperature (in units of J/k_B).
    dT : float
        Temperature decrement between successive points.
    seed : int or None
        PRNG seed for reproducibility. None uses the current time.

    Returns
    -------
    pd.DataFrame
        One row per temperature, sorted ascending, with columns:

        T   -- temperature
        M   -- mean absolute magnetisation per spin, <|M|> / N
        E   -- mean energy per spin, <E> / N  (≈ -2 in the ground state)
        Cv  -- specific heat per spin, (<E²> - <E>²) / (N T²)
        chi -- magnetic susceptibility per spin, (<M²> - <M>²) / (N T)
    """
    _seed = int(time.time()) % (2**31) if seed is None else int(seed)

    N = L * L
    n_equil = max(1, int(0.1 * MCS))  # sweeps to discard for equilibration

    key = random.PRNGKey(_seed)

    # Fully ordered initial state
    S = jnp.ones((L, L), dtype=jnp.int8)

    # Precompute checkerboard sublattice masks (constants, embedded in XLA graph)
    ii, jj = jnp.meshgrid(jnp.arange(L), jnp.arange(L), indexing="ij")
    mask_A = (ii + jj) % 2 == 0   # sublattice A: (i+j) even
    mask_B = ~mask_A               # sublattice B: (i+j) odd

    # ------------------------------------------------------------------
    # Pure helper functions (no side effects — JAX requirement)
    # ------------------------------------------------------------------

    def _calc_energy(S: jax.Array) -> jax.Array:
        """Bond energy E = sum_{bonds} s_i s_j, each bond counted once."""
        S32 = S.astype(jnp.int32)
        return jnp.sum(S32 * (jnp.roll(S32, -1, axis=0) + jnp.roll(S32, -1, axis=1)))

    def _half_sweep(S: jax.Array, mask: jax.Array, T: jax.Array, key: jax.Array) -> jax.Array:
        """
        Metropolis update for one checkerboard sublattice.

        All sites in `mask` are updated simultaneously. Returns new S
        (immutable — uses jnp.where instead of in-place mutation).
        """
        S16 = S.astype(jnp.int16)
        nb = (
            jnp.roll(S16, 1,  axis=0)
            + jnp.roll(S16, -1, axis=0)
            + jnp.roll(S16, 1,  axis=1)
            + jnp.roll(S16, -1, axis=1)
        )
        # Local field energy: s_i * sum(neighbours), values in {-4,-2,0,2,4}
        Senergy = S16 * nb

        # Metropolis acceptance: P = min(1, exp(-2*Senergy/T))
        # When Senergy <= 0 (energy decreases on flip) -> always accept.
        # Use jnp.where to avoid computing exp for non-negative exponents
        # (which would be >= 1 and overflow-prone at low T).
        exponent = -2.0 * Senergy.astype(jnp.float32) / T
        accept = jnp.where(exponent >= 0.0, 1.0, jnp.exp(exponent))

        rand = random.uniform(key, (L, L), dtype=jnp.float32)
        flip = mask & (rand < accept)

        # Immutable conditional flip: +1 -> -1 or -1 -> +1 where flip is True
        return jnp.where(flip, -S, S).astype(jnp.int8)



    @jit
    def _temperature_step(
        S: jax.Array,
        key: jax.Array,
        T: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run MCS sweeps at temperature T. Returns (S_new, key_new, Es, Ms)."""

        def sweep(carry: tuple, _: None) -> tuple:
            S, key = carry
            key, k1, k2 = random.split(key, 3)
            S = _half_sweep(S, mask_A, T, k1)
            S = _half_sweep(S, mask_B, T, k2)
            # Record observables after each sweep
            E = _calc_energy(S)
            M = jnp.sum(S, dtype=jnp.int32)
            return (S, key), (E, M)

        (S_new, key_new), (Es, Ms) = lax.scan(sweep, (S, key), None, length=MCS)
        return S_new, key_new, Es, Ms

    # ------------------------------------------------------------------
    # Temperature sweep
    # ------------------------------------------------------------------
    results = []
    T_py = round(T_start, 8)

    while T_py >= T_end - 1e-10:
        T_jax = jnp.float32(T_py)
        S, key, Es, Ms = _temperature_step(S, key, T_jax)

        # Discard equilibration sweeps and compute statistics in float64
        Es_eq = Es[n_equil:].astype(jnp.float64)
        Ms_eq = Ms[n_equil:].astype(jnp.float64)

        Eavg  = float(jnp.mean(Es_eq))
        E2avg = float(jnp.mean(Es_eq ** 2))
        Savg  = float(jnp.mean(Ms_eq))
        S2avg = float(jnp.mean(Ms_eq ** 2))
        Mavg  = float(jnp.mean(jnp.abs(Ms_eq))) / N
        Cv    = (E2avg - Eavg  ** 2) / (N * T_py ** 2)
        chi   = (S2avg - Savg  ** 2) / (N * T_py)

        results.append({"T": T_py, "M": Mavg, "E": -Eavg / N, "Cv": Cv, "chi": chi})
        T_py = round(T_py - dT, 8)

    return pd.DataFrame(results).sort_values("T").reset_index(drop=True)
