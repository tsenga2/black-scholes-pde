% Clear workspace and figures
clc
clear all
close all

% Define initial constants
N = 4000;           % Number of time steps
M = 1000;           % Number of stock price steps
Smin = 0.4;        % Minimum stock price (>0 for valid log)
Smax = 1000;        % Maximum stock price
T = 1;             % Time to maturity (1 year)
K = 10;            % Strike price
volatility = 0.4;  % Volatility (40%)
r = 0.02;          % Risk-free rate (2%)
d = 0;             % Dividend yield
is_call = 1;       % Flag for call option

% Initialize grid
surface = zeros(1+N, 1+M);  % Option price surface
dt = T/N;               % Time step size
dS = (Smax-Smin)/M;     % Stock price step size
t_vals = 0:dt:T;        % Time grid
S_vals = Smin:dS:Smax;  % Stock price grid

% Set boundary conditions
surface(:,1) = 0;          % At S = Smin (left boundary)
surface(:,end) = Smax-K;   % At S = Smax (right boundary)
surface(end,:) = payoff(S_vals, K, is_call);  % At t = T (terminal condition)

% Define tridiagonal matrix coefficients
a = @(j) 0.5*(r-d)*j*dt - 0.5*volatility.^2.*j.^2*dt;    % Lower diagonal
b = @(j) 1 + volatility.^2.*j.^2*dt + r*dt;              % Main diagonal
c = @(j) -0.5*(r-d)*j*dt - 0.5*volatility.^2.*j.^2*dt;   % Upper diagonal

% Solve backwards in time
for i = N:-1:1
    % Build tridiagonal matrix
    A = diag(a(2:M-1),-1) + diag(b(2:M)) + diag(c(1:M-2),1);
    
    % Set up RHS vector with boundary adjustments
    v = surface(i+1,2:M)';
    v(1) = v(1) - a(1)*surface(i,1);
    v(end) = v(end) - c(M+1)*surface(i,M+1);
    
    % Solve system
    surface(i,2:M) = A\v;
    surface(i,2:M) = max(surface(i,2:M),payoff(S_vals(2:M),K,is_call));
end

% interpolate to check surface
S_check = linspace(0,1.5*K,20);
t_check = linspace(0,T,20);

surf_check = zeros(size(S_check,2),size(t_check,2));
for i=1:size(t_check,2)
    for j=1:size(S_check,2)
        surf_check(i,j) = interp2(S_vals,t_vals,surface,S_check(j),t_check(i));
    end
end

% Plotting


%% Plot surface
surf(S_check,fliplr(t_check),surf_check)
xlabel('Stock Price')
ylabel('Time Until Maturity')
title('Example Surface')


figure('Position', [100 100 800 400]);
hold on;
plot(S_check, surf_check(1,:), 'b-', 'LineWidth', 2, 'DisplayName', 't = 0 (Initial)');
plot(S_check, surf_check(end,:), 'r--', 'LineWidth', 2, 'DisplayName', 't = T (Terminal)');
plot([K K], [0 max(surf_check(1,:))], 'k--', 'LineWidth', 1, 'DisplayName', 'Strike Price');

% Plot formatting
xlabel('Stock Price (S)');
ylabel('Option Price (V)');
title('Call Option Price vs Stock Price');
legend('show');
grid on;

% Payoff function definition
function out = payoff(S, K, is_call)
    if is_call
        out = max(S-K, 0);  % Call option payoff
    else
        out = max(K-S, 0);  % Put option payoff
    end
end