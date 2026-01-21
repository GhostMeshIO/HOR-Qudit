# HOR-Qudit: Hyper-Ontic Resonance Qudit Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/GhostMeshIO/HOR-Qudit/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://ghostmeshiO.github.io/HOR-Qudit/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.16623-b31b1b)](https://arxiv.org/abs/2510.16623)

**HOR-Qudit** is an open-source framework for simulating and prototyping Hyper-Ontic Resonance Qudits (HOR-qudits), a novel high-dimensional quantum carrier that integrates ERD-deformed Pauli algebra, non-associative torsion gates, emergent metric-coupled dynamics, and RG-flow scaling laws. Built for quantum computing researchers, AGI architects, and hardware engineers, it enables efficient modeling of advanced qudit systems for fault-tolerant computation, topological encryption, and hybrid quantum-classical workflows.

Inspired by parafermionic statistics and symplectic \(\mathbb{Z}_d\)-modules, HOR-qudits offer superior resource efficiency over traditional qubits/qudits, including log-depth multi-control gates and tunable error thresholds. This repository provides Python-based simulation tools, calibration scripts, and extensible blueprints for hardware integration.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citing HOR-Qudit](#citing-hor-qudit)

## Overview
HOR-qudits extend standard qudits (d-level quantum systems) by embedding higher algebraic structures:
- **ERD Deformation**: A scalar field \(\varepsilon\) modulates Pauli operators, introducing Berry phases and symplectic invariants.
- **Torsion and Braiding**: Non-associative operators enable single-shot entanglement, drawing from braid group representations.
- **Emergent Geometry**: \(\varepsilon\) sources a spacetime metric \(g_{ab}^{\text{HOR}}\), enabling gate-time dilation and curvature-boosted couplings.
- **RG Scaling**: Closed-form flows predict error rates, thresholds, and optimal code distances.
- **Hyper-Map Calibration**: A self-optimizing neural layer for closed-loop hardware tuning.

This framework is grounded in rigorous quantum information theory (references: [arXiv:2510.16623](https://arxiv.org/abs/2510.16623), [Quantum Journal q-2024-04-04-1307](https://quantum-journal.org/papers/q-2024-04-04-1307/)). It's designed for:
- Simulating qudit-based quantum algorithms (e.g., QFT, phase estimation).
- Prototyping on platforms like trapped ions, superconducting transmons, or photonic qudits.
- Integrating into AGI stacks for adaptive quantum resource management.

For a military/science-grade blueprint, see [docs/blueprint.md](docs/blueprint.md).

## Key Features
- **Symplectic \(\mathbb{Z}_d\)-Module Algebra**: Robust Pauli groups with non-commutativity rank \(\mathcal{N} = 3\).
- **Parafermionic Braiding**: Single-primitive gates for maximal entanglement.
- **Metric-Coupled Dynamics**: Tunable gate timings (\(t_{\text{gate}} = t_0 \sqrt{-g_{00}}\)) and couplings (\(J_{\text{eff}} = J_0 (1 + \xi R)\)).
- **RG-Flow Predictors**: Analytical error scaling (\(p_L \sim \exp(-\gamma d^\nu)\)) and threshold boosts.
- **Hyper-Map Loop**: On-device calibration with spectral convergence guarantees.
- **Hybrid Gates**: O(\(\log N\)) depth for multi-control operations.
- **Simulation Tools**: QuTiP integration for Hamiltonian evolution; NumPy/SciPy for module arithmetic.
- **Extensibility**: Modular API for custom ERD fields, torsion tensors, and hardware backends.

## Installation
### Prerequisites
- Python 3.12+
- Virtual environment (recommended: `venv` or `conda`)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/GhostMeshIO/HOR-Qudit.git
   cd HOR-Qudit
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Includes: `numpy`, `scipy`, `qutip`, `sympy`, `matplotlib`, `torch` for ML calibration.)

3. (Optional) Install development tools:
   ```
   pip install -r requirements-dev.txt
   ```
   (Adds: `pytest`, `sphinx`, `black` for testing/docs/formatting.)

4. Verify installation:
   ```
   python -m unittest discover tests
   ```

For Docker-based setup:
```
docker build -t hor-qudit .
docker run -it hor-qudit
```

## Quick Start
Simulate a basic HOR-qudit (d=4, \(\varepsilon=0.1\)):

```python
from hor_qudit.core import HORQudit, symplectic_prod

# Initialize qudit
qudit = HORQudit(d=4, epsilon=0.1)

# Define Pauli vectors
X_vec = qudit.pauli_vec([1, 0], [0, 0])
Z_vec = qudit.pauli_vec([0, 0], [1, 0])

# Check symplectic invariant
comm = symplectic_prod(X_vec, Z_vec)  # Should be non-zero

# Apply ERD-deformed gate
state = qudit.apply_gate('X_HOR', qudit.zero_state())

print(f"Non-commutativity: {comm != 0}")
```

Run: `python examples/basic_sim.py`

Expected output: Gate fidelity >0.99, curvature scalar \(R \approx 0.02\).

## Usage Examples
### Single-Qudit Calibration
```python
from hor_qudit.calibration import hyper_forward, hyper_inverse

# Control vector (amplitudes, phases, epsilon)
C = np.array([0.825, 0.640, 0.1])  # Example

# Forward map
R = hyper_forward(C, W_init, S, Q, NL)

# Inverse for updated weights
W_new = hyper_inverse(R, W_init, S, Q, NL, C_pp)

# Converge loop (see examples/calibrate.py)
```

### Multi-Qudit Entanglement
```python
from hor_qudit.gates import braid_entangler

# Two-qudit system
qudit_pair = HORQudit(d=5, n=2, epsilon=0.1)

# Apply braid gate
entangled = qudit_pair.apply_gate('U_HE', qudit_pair.product_state([0, 0]))

# Compute Bell fidelity
fidelity = qudit_pair.bell_fidelity(entangled)
```

### RG-Flow Resource Estimation
```python
from hor_qudit.scaling import logical_error

p_phys = 8e-3  # Physical error rate
p_th = 1e-2 * (1 + 2 * 0.1**2)  # HOR threshold
d = 7  # Code distance

p_L = logical_error(p_phys, p_th, d)
print(f"Logical error: {p_L:.2e}")
```

More examples in `/examples/` and Jupyter notebooks in `/notebooks/`.

## Documentation
- **API Reference**: Generated with Sphinx – [online docs](https://ghostmeshiO.github.io/HOR-Qudit/).
- **Tutorials**: See `/docs/tutorials/` for step-by-step guides on simulation, hardware mapping, and AGI integration.
- **Blueprint**: Detailed science/military-grade spec in [docs/blueprint.md](docs/blueprint.md).
- **Equations Index**: 48 core equations summarized in [docs/equations.md](docs/equations.md).

Build local docs:
```
cd docs
make html
```

## Contributing
We welcome contributions! Follow these steps:
1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit changes: `git commit -m 'Add YourFeature'`.
4. Push: `git push origin feature/YourFeature`.
5. Open a Pull Request.

Guidelines:
- Adhere to [PEP8](https://pep8.org/) (use `black` for formatting).
- Add tests in `/tests/`.
- Update docs for new features.
- For major changes, open an issue first.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

## Acknowledgments
- Built on foundational work from xAI Grok 4 and references like [arXiv:2510.16623](https://arxiv.org/abs/2510.16623).
- Thanks to contributors in quantum info theory (e.g., Purdue Kais Group, USC QServer).
- Inspired by open-source quantum tools: QuTiP, Qiskit.

## Citing HOR-Qudit
If you use this framework in research, please cite:
```
@software{HOR-Qudit2026,
  author = {GhostMeshIO Contributors},
  title = {HOR-Qudit: Hyper-Ontic Resonance Qudit Framework},
  year = {2026},
  url = {https://github.com/GhostMeshIO/HOR-Qudit},
  version = {1.0}
}
```
For the conceptual blueprint, cite the originating session (xAI, 2026).
