# Black-Scholes Option Pricing Model: Numerical Solutions
## Introduction to Financial Option Pricing

This notebook explores the numerical solutions to the Black-Scholes partial differential equation (PDE) for option pricing. We'll implement both explicit and implicit finite difference methods using Python.

## 1. The Black-Scholes Model

The Black-Scholes PDE is given by:

$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + (r-d)S\frac{\partial V}{\partial S} - rV = 0$

Where:
- $V(S,t)$ is the option price
- $S$ is the stock price
- $t$ is time
- $\sigma$ is volatility
- $r$ is risk-free interest rate
- $d$ is dividend yield

## 2. Implementation in Python

First, let's import the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def payoff(S, K, is_call=True):
    """Calculate the payoff of an option"""
    if is_call:
        return np.maximum(S - K, 0)
    return np.maximum(K - S, 0)
```

### 2.1 Implicit Finite Difference Method

```python
def black_scholes_implicit(N, M, Smin, Smax, T, K, sigma, r, d, is_call=True):
    """
    Solve Black-Scholes PDE using implicit finite difference method
    
    Parameters:
    -----------
    N : int
        Number of time steps
    M : int
        Number of stock price steps
    Smin, Smax : float
        Min and max stock prices
    T : float
        Time to maturity
    K : float
        Strike price
    sigma : float
        Volatility
    r : float
        Risk-free rate
    d : float
        Dividend yield
    is_call : bool
        True for call option, False for put option
        
    Returns:
    --------
    t_vals : array
        Time grid points
    S_vals : array
        Stock price grid points
    surf : array
        Option prices surface
    """
    # Create grid
    dt = T/N
    dS = (Smax-Smin)/M
    
    t_vals = np.linspace(0, T, N+1)
    S_vals = np.linspace(Smin, Smax, M+1)
    
    # Initialize surface
    surf = np.zeros((N+1, M+1))
    
    # Set boundary conditions
    if is_call:
        surf[:, 0] = 0
        surf[:, -1] = Smax - K
        surf[-1, :] = payoff(S_vals, K, is_call)
    else:
        surf[:, 0] = K
        surf[:, -1] = 0
        surf[-1, :] = payoff(S_vals, K, is_call)
    
    # Create coefficient functions
    j = np.arange(1, M)
    a = lambda j: 0.5*(r-d)*j*dt - 0.5*sigma**2*j**2*dt
    b = lambda j: 1 + sigma**2*j**2*dt + r*dt
    c = lambda j: -0.5*(r-d)*j*dt - 0.5*sigma**2*j**2*dt
    
    # Create tridiagonal matrix
    A = np.diag(a(j[1:]), -1) + np.diag(b(j)) + np.diag(c(j[:-1]), 1)
    
    # Solve backwards in time
    for i in range(N-1, -1, -1):
        v = surf[i+1, 1:-1]
        v[0] = v[0] - a(1)*surf[i, 0]
        v[-1] = v[-1] - c(M-1)*surf[i, -1]
        
        surf[i, 1:-1] = np.linalg.solve(A, v)
        
        # Apply free boundary condition
        surf[i, 1:-1] = np.maximum(surf[i, 1:-1], 
                                 payoff(S_vals[1:-1], K, is_call))
    
    return t_vals, S_vals, surf
```

### 2.2 Explicit Finite Difference Method

```python
def black_scholes_explicit(N, M, Smin, Smax, T, K, sigma, r, d, is_call=True):
    """
    Solve Black-Scholes PDE using explicit finite difference method
    """
    # Create grid
    dt = T/N
    dS = (Smax-Smin)/M
    
    t_vals = np.linspace(0, T, N+1)
    S_vals = np.linspace(Smin, Smax, M+1)
    
    # Initialize surface
    surf = np.zeros((N+1, M+1))
    
    # Set boundary conditions
    if is_call:
        surf[:, 0] = 0
        surf[:, -1] = Smax - K
        surf[-1, :] = payoff(S_vals, K, is_call)
    else:
        surf[:, 0] = K
        surf[:, -1] = 0
        surf[-1, :] = payoff(S_vals, K, is_call)
    
    # Create coefficient functions
    j = np.arange(1, M)
    a = lambda j: 1/(1+r*dt)*(-0.5*(r-d)*j*dt + 0.5*sigma**2*j**2*dt)
    b = lambda j: 1/(1+r*dt)*(1 - sigma**2*j**2*dt)
    c = lambda j: 1/(1+r*dt)*(0.5*(r-d)*j*dt + 0.5*sigma**2*j**2*dt)
    
    # Solve backwards in time
    for i in range(N-1, -1, -1):
        for j in range(1, M):
            surf[i,j] = (a(j)*surf[i+1,j-1] + 
                        b(j)*surf[i+1,j] + 
                        c(j)*surf[i+1,j+1])
        
        # Apply free boundary condition
        surf[i, 1:-1] = np.maximum(surf[i, 1:-1], 
                                 payoff(S_vals[1:-1], K, is_call))
    
    return t_vals, S_vals, surf
```

## 3. Visualization and Analysis

Let's create a function to plot the results:

```python
def plot_option_surface(t_vals, S_vals, surf, title="Option Price Surface"):
    """Plot the option price surface"""
    T, S = np.meshgrid(t_vals, S_vals)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf_plot = ax.plot_surface(S, T, surf.T, cmap='viridis')
    
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Price')
    ax.set_title(title)
    
    plt.colorbar(surf_plot)
    plt.show()

# Example usage
# Parameters
N = 4000  # time steps
M = 1000  # stock price steps
Smin = 0.4
Smax = 1000
T = 1
K = 10
sigma = 0.4
r = 0.02
d = 0
is_call = True

# Solve using implicit method
t_vals, S_vals, surf_implicit = black_scholes_implicit(
    N, M, Smin, Smax, T, K, sigma, r, d, is_call
)

# Plot results
plot_option_surface(t_vals, S_vals, surf_implicit, 
                   "Black-Scholes Option Price (Implicit Method)")
```

## 4. Key Differences Between Methods

### Implicit Method
- Unconditionally stable
- Requires solving a linear system at each time step
- More computationally intensive per step
- Can use larger time steps

### Explicit Method
- Conditionally stable (requires CFL condition)
- Simple to implement
- Less computationally intensive per step
- Requires smaller time steps for stability

## 5. Stability Analysis

The stability of the explicit method is governed by the CFL (Courant-Friedrichs-Lewy) condition:

$\Delta t \leq \frac{(\Delta S)^2}{(\sigma S)^2}$

For our implementation, we should check:
```python
def check_stability(dS, dt, S, sigma):
    """Check if the explicit method is stable"""
    return dt <= (dS**2)/(sigma*S)**2

# Example check
dS = (Smax-Smin)/M
dt = T/N
S_mid = (Smax+Smin)/2
is_stable = check_stability(dS, dt, S_mid, sigma)
print(f"Explicit method stability at midpoint: {'Stable' if is_stable else 'Unstable'}")
```

## 6. Exercises

1. Compare the results of explicit and implicit methods for different parameter values.
2. Analyze the convergence rate as you increase the number of grid points.
3. Implement the Crank-Nicolson method, which is a combination of explicit and implicit methods.
4. Add Greeks calculations (Delta, Gamma, Theta) to the implementation.

## References

1. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637-654.
2. Wilmott, P. (2006). Paul Wilmott on Quantitative Finance, 2nd Edition. Wiley.
