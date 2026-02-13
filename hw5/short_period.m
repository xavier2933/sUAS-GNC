clear, clc, close all
addpath('..');

vars = [18;0;1800];
params = utils.ttwistor();
time = 25; % seconds

[x0,u0] = utils.CalculateTrim(vars, params)
[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params);

[evec_lon, eval_lon_diag] = eig(Alon);
eigenvalues = diag(eval_lon_diag);


complex_indices = find(imag(eigenvalues) > 0.01);
[~, min_idx] = max(abs(eigenvalues(complex_indices)));
phugoid_idx = complex_indices(min_idx);

v_phugoid = evec_lon(:, phugoid_idx)
theta_idx = 4; 
target_theta_rad = deg2rad(2);

scale_factor = target_theta_rad / v_phugoid(theta_idx);
v_scaled = real(v_phugoid * scale_factor);

u_bar     = real(v_scaled(1));
alpha_bar = real(v_scaled(2));
q_bar     = real(v_scaled(3));
theta_bar = real(v_scaled(4));
h_bar     = real(v_scaled(5));

Va_star = 18;
theta_star = x0(5);
u_star = x0(7);
w_star = x0(9);

alpha_star = atan2(w_star, u_star); 

w_bar = (Va_star * cos(alpha_star)) * alpha_bar;

sys_lon = ss(Alon, zeros(5,1), eye(5), zeros(5,1));

% Simulate the response to the initial perturbation
[y_lin_perturbation, t_lin] = initial(sys_lon, v_scaled, time);

x_lin_full = repmat(x0, 1, length(t_lin));

x_lin_full(7, :)  = x0(7)  + y_lin_perturbation(:, 1)'; % u = u_star + u_bar
x_lin_full(11, :) = x0(11) + y_lin_perturbation(:, 3)'; % q = q_star + q_bar
x_lin_full(5, :)  = x0(5)  + y_lin_perturbation(:, 4)'; % theta = theta_star + theta_bar
x_lin_full(3, :)  = x0(3)  - y_lin_perturbation(:, 5)'; % z = z_star - h_bar

% Convert alpha_bar(t) to w(t) using the transformation w_bar = Va*cos(alpha)*alpha_bar
% w = w_star + w_bar
w_bar_t = (Va_star * cos(alpha_star)) * y_lin_perturbation(:, 2)';
x_lin_full(9, :)  = x0(9)  + w_bar_t;

u_lin_total = x_lin_full(7, :);
delta_x_lin = cumtrapz(t_lin, u_lin_total);
x_lin_full(1, :) = x0(1) + delta_x_lin;

u_lin = repmat(u0, 1, length(t_lin));


x0_sim = x0;
x0_sim(7)  = u_star + u_bar;     % Total u
x0_sim(9)  = w_star + w_bar;     % Total w
x0_sim(11) = x0(11) + q_bar;     % Total q
x0_sim(5)  = theta_star + theta_bar; % Total theta

x0_sim(3)  = x0(3) - h_bar;

options = odeset('MaxStep', 0.1);
[t_nonlin, x_nonlin_raw] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, [0;0;0], params), [0 time], x0_sim, options);
x_nonlin = x_nonlin_raw'; 

u_nonlin = repmat(u0, 1, length(t_nonlin));

utils.PlotSimulation(t_lin, x_lin_full, u_lin, 'r');
utils.PlotSimulation(t_nonlin, x_nonlin, u_nonlin, 'b'); 

