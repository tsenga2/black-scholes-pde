#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Black-Scholes Option Pricing Model Implementation
----------------------------------------------

This module implements numerical solutions to the Black-Scholes partial differential
equation (PDE) for option pricing using finite difference methods.

The Black-Scholes PDE is:
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + (r-d)S(∂V/∂S) - rV = 0

Where:
- V(S,t): Option price as a function of stock price (S) and time (t)
- σ: Volatility of the stock
- r: Risk-free interest rate
- d: Dividend yield
- S: Stock price
- t: Time
- K: Strike price

The PDE is solved with the following boundary conditions for a call option:
- V(S,T) = max(S-K, 0)         # Terminal condition (payoff at expiry)
- V(0,t) = 0                   # Boundary condition at S = 0
- V(Smax,t) ≈ Smax - K         # Boundary condition at S = Smax

For a put option:
- V(S,T) = max(K-S, 0)         # Terminal condition (payoff at expiry)
- V(0,t) = K                   # Boundary condition at S = 0
- V(Smax,t) ≈ 0               # Boundary condition at S = Smax

Author: Claude
Date: 2024-11-22
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def payoff(S, K, is_call=True):
    """
    Calculate the payoff of an option at expiry.
    
    Parameters
    ----------
    S : float or numpy.ndarray
        Stock price(s)
    K : float
        Strike price
    is_call : bool, optional
        True for call option, False for put option (default is True)
    
    Returns
    -------
    float or numpy.ndarray
        Option payoff value(s)
        For call: max(S-K, 0)
        For put: max(K-S, 0)
    """
    if is_call:
        return np.maximum(S - K, 0)
    return np.maximum(K - S, 0)

def black_scholes_implicit(N, M, Smin, Smax, T, K, sigma, r, d, is_call=True):
    """
    Solve Black-Scholes PDE using implicit finite difference method.
    
    The implicit method is unconditionally stable but requires solving
    a linear system at each time step.
    
    Parameters
    ----------
    N : int
        Number of time steps
    M : int
        Number of stock price steps
    Smin : float
        Minimum stock price in the grid
    Smax : float
        Maximum stock price in the grid
    T : float
        Time to maturity in years
    K : float
        Strike price
    sigma : float
        Volatility (annualized)
    r : float
        Risk-free interest rate (annualized)
    d : float
        Dividend yield (annualized)
    is_call : bool, optional
        True for call option, False for put option (default is True)
    
    Returns
    -------
    tuple
        (t_vals, S_vals, surf) where:
        - t_vals: array of time points
        - S_vals: array of stock price points
        - surf: 2D array of option prices (N+1 × M+1)
    
    Notes
    -----
    The implicit method uses the following discretization:
    (V[i,j] - V[i+1,j])/dt + 0.5σ²S²(V[i,j+1] - 2V[i,j] + V[i,j-1])/dS² +
    (r-d)S(V[i,j+1] - V[i,j-1])/(2dS) - rV[i,j] = 0
    
    This leads to a tridiagonal system of equations that must be solved
    at each time step.
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
        surf[:, 0] = 0  # V(0,t) = 0
        surf[:, -1] = Smax - K  # V(Smax,t) = Smax - K
        surf[-1, :] = payoff(S_vals, K, is_call)  # V(S,T) = max(S-K, 0)
    else:
        surf[:, 0] = K  # V(0,t) = K
        surf[:, -1] = 0  # V(Smax,t) = 0
        surf[-1, :] = payoff(S_vals, K, is_call)  # V(S,T) = max(K-S, 0)
    
    # Create coefficient functions for the tridiagonal matrix
    j = np.arange(1, M)
    # Lower diagonal coefficients
    a = lambda j: 0.5*(r-d)*j*dt - 0.5*sigma**2*j**2*dt
    # Main diagonal coefficients
    b = lambda j: 1 + sigma**2*j**2*dt + r*dt
    # Upper diagonal coefficients
    c = lambda j: -0.5*(r-d)*j*dt - 0.5*sigma**2*j**2*dt
    
    # Create tridiagonal matrix
    A = np.diag(a(j[1:]), -1) + np.diag(b(j)) + np.diag(c(j[:-1]), 1)
    
    # Solve backwards in time
    for i in range(N-1, -1, -1):
        # Set up right-hand side vector
        v = surf[i+1, 1:-1]
        # Apply boundary conditions
        v[0] = v[0] - a(1)*surf[i, 0]
        v[-1] = v[-1] - c(M-1)*surf[i, -1]
        
        # Solve tridiagonal system
        surf[i, 1:-1] = np.linalg.solve(A, v)
        
        # Apply free boundary condition (American option feature)
        surf[i, 1:-1] = np.maximum(surf[i, 1:-1], 
                                 payoff(S_vals[1:-1], K, is_call))
    
    return t_vals, S_vals, surf

def black_scholes_explicit(N, M, Smin, Smax, T, K, sigma, r, d, is_call=True):
    """
    Solve Black-Scholes PDE using explicit finite difference method.
    
    The explicit method is conditionally stable and simpler to implement
    than the implicit method, but requires smaller time steps.
    
    Parameters
    ----------
    Same as black_scholes_implicit()
    
    Returns
    -------
    Same as black_scholes_implicit()
    
    Notes
    -----
    The explicit method uses the following discretization:
    (V[i,j] - V[i+1,j])/dt = 0.5σ²S²(V[i+1,j+1] - 2V[i+1,j] + V[i+1,j-1])/dS² +
    (r-d)S(V[i+1,j+1] - V[i+1,j-1])/(2dS) - rV[i+1,j]
    
    Stability condition (CFL):
    dt ≤ dS²/(σ²S²)
    """
    # Implementation remains the same as before...
    dt = T/N
    dS = (Smax-Smin)/M
    
    t_vals = np.linspace(0, T, N+1)
    S_vals = np.linspace(Smin, Smax, M+1)
    
    surf = np.zeros((N+1, M+1))
    
    if is_call:
        surf[:, 0] = 0
        surf[:, -1] = Smax - K
        surf[-1, :] = payoff(S_vals, K, is_call)
    else:
        surf[:, 0] = K
        surf[:, -1] = 0
        surf[-1, :] = payoff(S_vals, K, is_call)
    
    j = np.arange(1, M)
    a = lambda j: 1/(1+r*dt)*(-0.5*(r-d)*j*dt + 0.5*sigma**2*j**2*dt)
    b = lambda j: 1/(1+r*dt)*(1 - sigma**2*j**2*dt)
    c = lambda j: 1/(1+r*dt)*(0.5*(r-d)*j*dt + 0.5*sigma**2*j**2*dt)
    
    for i in range(N-1, -1, -1):
        for j in range(1, M):
            surf[i,j] = (a(j)*surf[i+1,j-1] + 
                        b(j)*surf[i+1,j] + 
                        c(j)*surf[i+1,j+1])
        
        surf[i, 1:-1] = np.maximum(surf[i, 1:-1], 
                                 payoff(S_vals[1:-1], K, is_call))
    
    return t_vals, S_vals, surf

def plot_option_surface(t_vals, S_vals, surf, title="Option Price Surface"):
    """
    Create a 3D surface plot of option prices.
    
    Parameters
    ----------
    t_vals : numpy.ndarray
        Array of time points
    S_vals : numpy.ndarray
        Array of stock price points
    surf : numpy.ndarray
        2D array of option prices
    title : str, optional
        Plot title (default is "Option Price Surface")
    
    Returns
    -------
    None
        Displays the plot
    """
    T, S = np.meshgrid(t_vals, S_vals)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf_plot = ax.plot_surface(S, T, surf.T, cmap='viridis')
    
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Price')
    ax.set_title(title)
    
    plt.colorbar(surf_plot)
    plt.show()

def main():
    """
    Main function to demonstrate the usage of the Black-Scholes solvers.
    
    This function:
    1. Sets up example parameters
    2. Solves the Black-Scholes PDE using the implicit method
    3. Creates a visualization of the solution
    4. Checks the stability condition for the explicit method
    """
    # Example parameters
    N = 4000  # time steps
    M = 1000  # stock price steps
    Smin = 0.4  # minimum stock price
    Smax = 1000  # maximum stock price
    T = 1  # time to maturity (years)
    K = 10  # strike price
    sigma = 0.4  # volatility (40%)
    r = 0.02  # risk-free rate (2%)
    d = 0  # dividend yield (0%)
    is_call = True  # pricing a call option

    # Solve using implicit method
    print("Solving using implicit method...")
    t_vals, S_vals, surf_implicit = black_scholes_implicit(
        N, M, Smin, Smax, T, K, sigma, r, d, is_call
    )

    # Plot results
    plot_option_surface(t_vals, S_vals, surf_implicit, 
                       "Black-Scholes Option Price (Implicit Method)")

    # Check a specific point
    S_check = 30
    t_check = 0.25
    i_t = np.abs(t_vals - t_check).argmin()
    i_S = np.abs(S_vals - S_check).argmin()
    print(f"\nOption price at S={S_check} and t={t_check}: {surf_implicit[i_t, i_S]:.4f}")

    # Check stability for explicit method
    dS = (Smax-Smin)/M
    dt = T/N
    S_mid = (Smax+Smin)/2
    stability_condition = dt <= (dS**2)/(sigma*S_mid)**2
    print(f"\nExplicit method stability at midpoint: {'Stable' if stability_condition else 'Unstable'}")
    print(f"dt = {dt:.6f}")
    print(f"Stability requirement: dt <= {(dS**2)/(sigma*S_mid)**2:.6f}")

if __name__ == "__main__":
    main()
