Teo Price-Broncucia

# Additional ODE-Net Examples

This library provides ordinary differential equation (ODE) solvers implemented in PyTorch. Backpropagation through all solvers is supported using the adjoint method. For usage of ODE solvers in deep learning applications, see [1].

As the solvers are implemented in PyTorch, algorithms in this repository are fully supported to run on the GPU.

---

## Installation
```
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install -e .
```

Add additional example files, latent_ode_2pend.py, ode_demo_1pend.py, and pendulum_sim.py.

Run examples by calling like

```
python ode_demo_1pend.py --viz
```

## Examples
Examples are placed in the [`examples`](./examples) directory.


### References
[1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018. [[arxiv]](https://arxiv.org/abs/1806.07366)
