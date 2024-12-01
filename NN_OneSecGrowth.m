% Main script: Solving HJB equation using Neural Networks
clear all; close all;

%% Declare global parameters
global sigma alpha delta rho A

%% Problem Definition
% HJB equation: max{V_k,ε}^((σ-1)/σ) + max{V_k,ε}[Ak^α - δk] - ρV = 0
% State variable: k (capital)
% Control variable: ε (consumption)

%% Economic Parameters
sigma = 2;      % Elasticity of intertemporal substitution
alpha = 0.33;   % Capital share
delta = 0.05;   % Depreciation rate
rho = 0.05;     % Discount rate
A = 1;          % Total factor productivity

%% Network Parameters
N = 7;         % Number of hidden neurons
learning_rate = 0.05;
max_its = 300000;
tol = 1e-6;

%% Initialize Network Parameters
% Random initialization with Xavier/Glorot initialization
rng(42); % For reproducibility
v = randn(N, 1) * sqrt(2/1);   % Input weights
theta = randn(N, 1) * sqrt(2/1);% Hidden biases
w = randn(N, 1) * sqrt(2/N);   % Output weights

%% Generate Training Points
nk = 50;
k_min = 0.1;    % Minimum capital
k_max = 10;     % Maximum capital
k_grid = linspace(k_min, k_max, nk)';

%% Training Loop
tic;
fprintf('Training Neural Network for HJB equation...\n');
for it = 1:max_its
    % Forward Pass to get value function and its derivative
    [V, Vk] = forward_pass(k_grid, v, theta, w);
    
    % Compute optimal consumption using FOC
    Vk_pos = max(Vk, 0.001);
    c_safe = Vk_pos.^(-1/sigma);
    
    % Compute HJB error
    production = A * k_grid.^alpha;
    hjb_error = (sigma/(1.0-sigma)) * c_safe.^((sigma-1)/sigma) + c_safe.*(production - delta*k_grid) - rho*V;
    
    % Compute loss
    loss = mean(hjb_error.^2);

    if it < 100
        fprintf('Iteration %d: Loss = %.6f\n', it, loss);
    end 
    if mod(it, 1000) == 0
        fprintf('Iteration %d: Loss = %.6f\n', it, loss);
    end
    
    if loss < tol
        break;
    end
    
    % Backpropagation
    [grad_v, grad_theta, grad_w] = compute_gradients(k_grid, v, theta, w, hjb_error);
    
    % Update Parameters
    v = v - learning_rate * grad_v;
    theta = theta - learning_rate * grad_theta;
    w = w - learning_rate * grad_w;
end
toc;

%% Evaluate Results
[V_final, Vk_final] = forward_pass(k_grid, v, theta, w);
c_star_final = Vk_final.^(-1/sigma);

% Plot Results
figure('Position', [100, 100, 1200, 800]);

% Value Function
subplot(2,2,1);
plot(k_grid, V_final, 'b-', 'LineWidth', 2);
xlabel('Capital (k)');
ylabel('Value Function V(k)');
title('Value Function');
grid on;

% Policy Function (Consumption)
subplot(2,2,2);
plot(k_grid, c_star_final, 'r-', 'LineWidth', 2);
xlabel('Capital (k)');
ylabel('Consumption (ε)');
title('Consumption Policy');
grid on;

% Value Function Derivative
subplot(2,2,3);
plot(k_grid, Vk_final, 'g-', 'LineWidth', 2);
xlabel('Capital (k)');
ylabel('V_k(k)');
title('Value Function Derivative');
grid on;

% HJB Error
subplot(2,2,4);
production = A * k_grid.^alpha;
hjb_error = (sigma/(1.0-sigma)) * c_star_final.^((sigma-1)/sigma) + c_star_final.*(production - delta*k_grid) - rho*V_final;
plot(k_grid, abs(hjb_error), 'k-', 'LineWidth', 2);
xlabel('Capital (k)');
ylabel('|HJB Error|');
title('HJB Error');
set(gca, 'YScale', 'log');
grid on;

%% Helper Functions
function [V, Vk] = forward_pass(k, v, theta, w)
    % Input dimensions:
    % k: [n_points × 1] - capital values
    % v: [N × 1] - input weights
    % theta: [N × 1] - bias terms
    % w: [N × 1] - output weights
    
    N = length(v);
    n_points = length(k);
    
    % Correct broadcasting for matrix operations
    k_broadcast = repmat(k, 1, N);       % [n_points × N]
    v_broadcast = repmat(v', n_points, 1);   % [n_points × N]
    theta_broadcast = repmat(theta', n_points, 1);  % [n_points × N]
    
    % Hidden layer computations
    z = v_broadcast .* k_broadcast + theta_broadcast;  % [n_points × N]
    %y = sigmoid(z);  % [n_points × N]
    y = tanh(z);  % Using tanh instead of sigmoid for better performance
    
    % Derivative computations
    %dydk = v_broadcast.* sigmoid_derivative(y);  % [n_points × N]
    dydk = v_broadcast .* (1 - y.^2);  % derivative of tanh
    
    % Output layer
    V = y * w;    % [n_points × 1]
    Vk = dydk * w;  % [n_points × 1]
end

function [grad_v, grad_theta, grad_w] = compute_gradients(k, v, theta, w, hjb_error)
    % Compute gradients using numerical differentiation
    epsilon = 1e-6;
    N = length(v);
    grad_v = zeros(size(v));
    grad_theta = zeros(size(theta));
    grad_w = zeros(size(w));
    
    % Gradient for v
    for i = 1:N
        v_plus = v; v_plus(i) = v_plus(i) + epsilon;
        v_minus = v; v_minus(i) = v_minus(i) - epsilon;
        grad_v(i) = mean(compute_hjb_error(k, v_plus, theta, w).^2 - ...
                        compute_hjb_error(k, v_minus, theta, w).^2) / (2*epsilon);
    end
    
    % Gradient for theta
    for i = 1:N
        theta_plus = theta; theta_plus(i) = theta_plus(i) + epsilon;
        theta_minus = theta; theta_minus(i) = theta_minus(i) - epsilon;
        grad_theta(i) = mean(compute_hjb_error(k, v, theta_plus, w).^2 - ...
                           compute_hjb_error(k, v, theta_minus, w).^2) / (2*epsilon);
    end
    
    % Gradient for w
    for i = 1:N
        w_plus = w; w_plus(i) = w_plus(i) + epsilon;
        w_minus = w; w_minus(i) = w_minus(i) - epsilon;
        grad_w(i) = mean(compute_hjb_error(k, v, theta, w_plus).^2 - ...
                        compute_hjb_error(k, v, theta, w_minus).^2) / (2*epsilon);
    end
end

function hjb_error = compute_hjb_error(k_grid, v, theta, w)
    % Helper function to compute HJB error for gradient calculation
    global sigma alpha delta rho A
    
    [V, Vk] = forward_pass(k_grid, v, theta, w);

    Vk_pos = max(Vk, 0.001);
    c_safe = Vk_pos.^(-1/sigma);
    production = A * k_grid.^alpha;
    hjb_error = (sigma/(1.0-sigma)) * c_safe.^((sigma-1)/sigma) + c_safe.*(production - delta*k_grid) - rho*V;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function dy = sigmoid_derivative(x)
    s = sigmoid(x);
    dy = s .* (1 - s);
end