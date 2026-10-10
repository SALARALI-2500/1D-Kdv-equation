"""
Baseline PINN for single KdV soliton equation.
Depends on: kdv_exact_and_sampling.py (exact_solution, sample_ic_points,
sample_bc_points, sample_collocation_points)

PDE: u_t + 6 u u_x + u_xxx = 0
"""

import torch
import torch.nn as nn

from kdv_exact_and_sampling import (
    exact_solution,
    sample_ic_points,
    sample_bc_points,
    sample_collocation_points,
    x_min, x_max, t_min, t_max,
    DTYPE,
)

torch.manual_seed(0)


# ----------------------------
# 1. Network architecture
# ----------------------------
class PINN(nn.Module):
    """
    Fully-connected network: input (x, t) -> output u.
    """
    def __init__(self, layers=(2, 64, 64, 64, 64, 1)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            # Xavier initialization matches tanh activations better than
            # PyTorch's Kaiming default, which is tuned for ReLU-style nets.
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


# ----------------------------
# 2. PDE residual via autograd
# ----------------------------
def pde_residual(model, x, t):
    """
    Computes u_t + 6*u*u_x + u_xxx at the given (x, t) points,
    using automatic differentiation through the network's own output.
    """
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                 create_graph=True)[0]

    residual = u_t + 6.0 * u * u_x + u_xxx
    return residual


# ----------------------------
# 3. Combined loss
# ----------------------------
def compute_loss(model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f):
    # IC loss: match exact solution at t=0
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)

    # BC loss: match exact solution at x_min, x_max
    u_bc_pred = model(x_bc, t_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)

    # PDE residual loss: interior collocation points
    res = pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(res ** 2)

    total_loss = loss_ic + loss_bc + loss_pde
    return total_loss, loss_ic.item(), loss_bc.item(), loss_pde.item()


# ----------------------------
# 4. Training loop
# ----------------------------
def train(model, epochs=5000, lr=1e-3, print_every=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fixed training data (regenerated once, reused every epoch)
    x_ic, t_ic, u_ic = sample_ic_points(n_ic=200)
    x_bc, t_bc, u_bc = sample_bc_points(n_bc=100)
    x_f, t_f = sample_collocation_points(n_f=10000)

    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, l_ic, l_bc, l_pde = compute_loss(
            model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f
        )
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | total {loss.item():.6e} | "
                  f"IC {l_ic:.3e} | BC {l_bc:.3e} | PDE {l_pde:.3e}")

    return history, (x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f)


# ----------------------------
# 4b. L-BFGS fine-tuning phase (run after Adam)
# ----------------------------
def train_lbfgs(model, data, max_iter=500):
    """
    Polishes the Adam-trained model with L-BFGS, which converges more
    precisely once already in a good region of the loss landscape.
    """
    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f = data

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter,
        history_size=50, line_search_fn="strong_wolfe"
    )

    iteration_counter = [0]

    def closure():
        optimizer.zero_grad()
        loss, l_ic, l_bc, l_pde = compute_loss(
            model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f
        )
        loss.backward()
        iteration_counter[0] += 1
        if iteration_counter[0] % 50 == 0:
            print(f"L-BFGS iter {iteration_counter[0]:5d} | total {loss.item():.6e} | "
                  f"IC {l_ic:.3e} | BC {l_bc:.3e} | PDE {l_pde:.3e}")
        return loss

    optimizer.step(closure)


# ----------------------------
# 5. Evaluation: relative L2 error against exact solution
# ----------------------------
def relative_l2_error(model, n_test=200):
    """
    Evaluates the trained model on a grid and compares against the exact
    solution, returning the relative L2 error:
        ||u_pred - u_exact||_2 / ||u_exact||_2
    """
    x_test = torch.linspace(x_min, x_max, n_test, dtype=DTYPE).reshape(-1, 1)
    t_test = torch.linspace(t_min, t_max, n_test, dtype=DTYPE).reshape(-1, 1)

    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing="ij")
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(X_flat, T_flat)
    u_exact = exact_solution(X_flat, T_flat)

    error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    return error.item()


# ----------------------------
# Run it
# ----------------------------
if __name__ == "__main__":
    model = PINN()

    print("=== Phase 1: Adam ===")
    history, data = train(model, epochs=5000, lr=1e-3, print_every=500)
    err_after_adam = relative_l2_error(model)
    print(f"Relative L2 error after Adam: {err_after_adam:.6e}")

    print("\n=== Phase 2: L-BFGS fine-tuning ===")
    train_lbfgs(model, data, max_iter=500)
    err_after_lbfgs = relative_l2_error(model)
    print(f"\nFinal relative L2 error after Adam + L-BFGS: {err_after_lbfgs:.6e}")
