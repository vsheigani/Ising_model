# 2D Ising Model — Monte Carlo Simulation

A Python implementation of the two-dimensional ferromagnetic Ising model, simulated with the Metropolis Monte Carlo algorithm. The code reproduces the paramagnetic-to-ferromagnetic phase transition and lets you measure thermodynamic observables (energy, magnetisation, specific heat, susceptibility) as functions of temperature.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Theory](#theory)
   - [The Ising Model](#the-ising-model)
   - [Thermodynamic Observables](#thermodynamic-observables)
   - [Phase Transition and Critical Temperature](#phase-transition-and-critical-temperature)
   - [Finite-Size Effects](#finite-size-effects)
3. [Numerical Method](#numerical-method)
   - [The Metropolis Algorithm](#the-metropolis-algorithm)
   - [Checkerboard Updates](#checkerboard-updates)
   - [Slow Cooling Protocol](#slow-cooling-protocol)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [Running the Simulation](#running-the-simulation)
8. [References](#references)

---

## Project Overview

The Ising model is one of the most important exactly solvable models in statistical mechanics. Despite its simplicity — binary spins on a lattice interacting only with their nearest neighbours — it captures the essential physics of second-order phase transitions, spontaneous symmetry breaking, and universality.

This project provides:

- **`ising_model.py`** — a self-contained Python module with a single function `run_ising_model(...)` that runs the simulation and returns results as a `pandas.DataFrame`.
- **`Ising_simulation.ipynb`** — a Jupyter notebook that runs the simulation and produces publication-quality plots of all thermodynamic observables, including a finite-size scaling comparison.

---

## Theory

### The Ising Model

Consider a 2D square lattice of $N = L \times L$ sites. Each site $i$ carries a spin variable $s_i \in \{-1, +1\}$ representing a magnetic moment pointing either up or down. The spins interact with their four nearest neighbours through the **Hamiltonian**:

$$
H = -J \sum_{\langle i,j \rangle} s_i\, s_j
$$

where:

- The sum runs over all nearest-neighbour pairs $\langle i,j \rangle$.
- $J > 0$ is the **exchange coupling constant** (ferromagnetic: aligned spins lower the energy).
- **Periodic boundary conditions** are applied in both directions, so every spin has exactly four neighbours and there are no edge effects.

We work in units where $J = 1$ and $k_B = 1$, making temperature dimensionless.

**Ground state** ($T = 0$): all spins aligned (either all $+1$ or all $-1$), with energy per spin $E/N = -2J = -2$.

**High-temperature limit** ($T \to \infty$): spins are completely random and uncorrelated, $E/N \to 0$.

### Thermodynamic Observables

In thermal equilibrium at temperature $T$, the system is described by the **canonical ensemble** with partition function $Z = \sum_{\{s\}} e^{-H/T}$. The observables we measure and their physical meaning:

| Observable | Formula | Physical meaning |
|---|---|---|
| Mean energy per spin | $\langle E \rangle / N$ | Average energy density |
| Magnetisation per spin | $\langle \|M\| \rangle / N$ | Strength of ferromagnetic order (order parameter) |
| Specific heat per spin | $C_v = (\langle E^2 \rangle - \langle E \rangle^2) / (N T^2)$ | Energy fluctuations; diverges at $T_c$ |
| Susceptibility per spin | $\chi = (\langle M^2 \rangle - \langle M \rangle^2) / (N T)$ | Response to external field; diverges at $T_c$ |

We use $|M|$ (absolute value of total magnetisation $M = \sum_i s_i$) rather than $M$ itself, because in a finite system with no external field, both ground states ($+1$ or $-1$ domains) are sampled equally, making the time-averaged $\langle M \rangle = 0$ even in the ordered phase.

The **fluctuation-dissipation relations** connect $C_v$ and $\chi$ to the fluctuations of $E$ and $M$, which is why they diverge at the critical point where fluctuations become largest.

### Phase Transition and Critical Temperature

The 2D Ising model undergoes a **continuous (second-order) phase transition** at the **Onsager critical temperature**, derived analytically by Lars Onsager in 1944:

$$
T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.2692\; J/k_B
$$

This is one of the very few exact results in statistical mechanics for an interacting many-body system.

**Below $T_c$** (ordered / ferromagnetic phase):
- $\langle |M| \rangle / N \to 1$ as $T \to 0$
- Long-range correlations between spins

**Above $T_c$** (disordered / paramagnetic phase):
- $\langle |M| \rangle / N \to 0$
- Exponential decay of spin-spin correlations

**At $T_c$**:
- Both $C_v$ and $\chi$ diverge (logarithmically and as a power law, respectively)
- Power-law decay of correlations — the system is **scale-invariant**
- Critical exponents: $\alpha = 0$ (log divergence of $C_v$), $\beta = 1/8$, $\gamma = 7/4$, $\nu = 1$

### Finite-Size Effects

A simulation can only access a finite lattice of size $L \times L$. This has several consequences:

- **Rounded transition**: the divergences of $C_v$ and $\chi$ are replaced by finite peaks.
- **Shifted pseudo-critical temperature**: the peak of $\chi$ occurs at a pseudo-critical temperature $T_c(L)$ that shifts toward $T_c$ as:
  $$T_c(L) \approx T_c + a\, L^{-1/\nu} = T_c + a\, L^{-1}$$
- **Finite peak heights**: the peak heights scale with $L$ as:
  $$C_v^{\max} \sim \ln L, \quad \chi^{\max} \sim L^{\gamma/\nu} = L^{7/4}$$

Studying how these peaks evolve with $L$ is called **finite-size scaling** and is the standard way to extract critical exponents numerically.

---

## Numerical Method

### The Metropolis Algorithm

The Metropolis algorithm generates a Markov chain of spin configurations that samples the Boltzmann distribution $P(\{s\}) \propto e^{-H/T}$ in equilibrium. One **Monte Carlo step** proceeds as follows:

1. Select a spin $s_i$ at random.
2. Compute the local field energy:
   $$\Delta E_{\text{flip}} = 2\, s_i \sum_{j \in \text{nn}(i)} s_j$$
   (the energy change if $s_i$ is flipped, derived from $H = -J \sum s_i s_j$).
3. Accept the flip with probability:
   $$P_{\text{accept}} = \min\!\left(1,\; e^{-\Delta E / T}\right)$$
   - If $\Delta E \leq 0$ (energy decreases): always accept.
   - If $\Delta E > 0$ (energy increases): accept with probability $e^{-\Delta E/T}$, which is large at high $T$ and small at low $T$.
4. Update $E$ and $M$ accordingly.

One **Monte Carlo sweep (MCS)** consists of $N = L^2$ such steps, so on average every spin is attempted once.

### Checkerboard Updates

Selecting spins uniformly at random (as in the original C++ code) requires a Python loop over $N$ spins per sweep, which is slow. Instead, we use **checkerboard (sublattice) updates**:

The lattice is divided into two sublattices — like the black and white squares of a chessboard:
- **Sublattice A**: sites $(i,j)$ with $(i+j)$ even
- **Sublattice B**: sites $(i,j)$ with $(i+j)$ odd

Crucially, all nearest neighbours of an A-site are B-sites and vice versa. This means **all sites within one sublattice can be updated simultaneously**: their acceptance decisions are independent because they do not interact with each other.

This allows fully vectorised NumPy operations, making the code orders of magnitude faster than a Python spin-by-spin loop while producing statistically equivalent results.

One sweep = one simultaneous update of sublattice A + one simultaneous update of sublattice B.

### Slow Cooling Protocol

Rather than re-initialising the lattice at each temperature, we carry the spin configuration over from one temperature step to the next. Starting from a fully ordered state at $T_{\text{start}} = 4.0$ and cooling slowly to $T_{\text{end}} = 0.5$, the system has time to equilibrate at each step. The first 10% of sweeps at each temperature are discarded as additional equilibration; the remaining 90% are used for averaging.

---

## Project Structure

```
Ising_model/
├── ising_model.py          # Python module: run_ising_model() function
├── Ising_simulation.ipynb  # Jupyter notebook: runs simulation and plots results
├── pyproject.toml          # Project dependencies (uv / pip)
└── README.md               # This file
```

---

## Setup

### Requirements

- Python 3.12+
- `numpy`
- `pandas`
- `matplotlib`
- `jupyter` / `notebook`

### Installation with `uv` (recommended)

```bash
# Clone or enter the project directory
cd Ising_model

# Install all dependencies into a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

---

## Running the Simulation

### As a Python module

```python
from ising_model import run_ising_model

# Run with default parameters (L=36, MCS=1000)
df = run_ising_model(L=36, MCS=1000, T_start=4.0, T_end=0.5, dT=0.05, seed=42)

# df is a pandas DataFrame with columns: T, M, E, Cv, chi
print(df.head())
```

**Function signature:**

```python
run_ising_model(
    L       = 36,    # Lattice size (L x L spins)
    MCS     = 1000,  # Monte Carlo sweeps per temperature
    T_start = 4.0,   # Starting temperature
    T_end   = 0.5,   # Ending temperature
    dT      = 0.05,  # Temperature step
    seed    = None,  # Random seed (None = non-deterministic)
) -> pd.DataFrame
```

**Increasing MCS** gives smoother, more accurate curves at the cost of longer runtime. Values of 2000–5000 are recommended for publication-quality results.

### As a Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook Ising_simulation.ipynb
# or
jupyter lab Ising_simulation.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Run a single simulation for $L = 36$, MCS = 1000.
2. Plot magnetisation, energy, specific heat, and susceptibility vs temperature.
3. Run simulations for $L \in {4, 12, 36, 100}$ and produce a finite-size scaling comparison.
4. Print a table of pseudo-critical temperatures and peak heights.

Saved plots (`observables_L36.png`, `finite_size_scaling.png`) will appear in the project directory.

---

## References

1. **Onsager, L.** (1944). "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition." *Physical Review*, 65(3–4), 117–149.
2. **Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E.** (1953). "Equation of State Calculations by Fast Computing Machines." *Journal of Chemical Physics*, 21(6), 1087.
3. **Newman, M. E. J., & Barkema, G. T.** (1999). *Monte Carlo Methods in Statistical Physics*. Oxford University Press.
4. **Landau, D. P., & Binder, K.** (2014). *A Guide to Monte Carlo Simulations in Statistical Physics* (4th ed.). Cambridge University Press.
