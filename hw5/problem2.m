clear, clc, close all
addpath('..');

vars = [18;0;1800];
params = utils.ttwistor();
time = 3; % seconds
control_idx = 1;
pulse_val = 3;
doublet = false;

[x0,u0] = utils.CalculateTrim(vars, params)

options = odeset('MaxStep', 0.1);
[t, x] = ode45(@(t, x) utils.AircraftEOMPulsed(t, x, u0, [0;0;0], params, control_idx, pulse_val, doublet), [0 time], x0, options);


u = repmat(u0, 1, length(t));
u(control_idx, t < 1) = u(control_idx, t < 1) + pulse_val;

utils.PlotSimulation(t, x', u, 'r');



[x0,u0] = utils.CalculateTrim(vars, params)
[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params);

[evec_lon, eval_lon_diag] = eig(Alon);
eigenvalues = diag(eval_lon_diag);


complex_indices = find(imag(eigenvalues) > 0.01);
[~, min_idx] = max(abs(eigenvalues(complex_indices)));
phugoid_idx = complex_indices(min_idx);

v_phugoid = evec_lon(:, phugoid_idx);
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

utils.PlotSimulation(t_nonlin, x_nonlin, u_nonlin, 'b'); 
