clear, clc, close all
addpath('..');

vars = [18;0;1800];
params = utils.ttwistor();

[x0,u0] = utils.CalculateTrim(vars, params)
[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params);

[evec_lon, eval_lon_diag] = eig(Alon);
eigenvalues = diag(eval_lon_diag);


complex_indices = find(imag(eigenvalues) > 0.01);
[~, min_idx] = min(abs(eigenvalues(complex_indices)));
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

sys_lon = ss(Alon, zeros(5,1), eye(5), zeros(5,1));

% Simulate the response to the initial perturbation
[y_lin_perturbation, t_lin] = initial(sys_lon, v_scaled, 250);

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
utils.PlotSimulation(t_lin, x_lin_full, u_lin, 'r');


x0_sim = x0;
x0_sim(7)  = u_star + u_bar;     % Total u
x0_sim(9)  = w_star + w_bar;     % Total w
x0_sim(11) = x0(11) + q_bar;     % Total q
x0_sim(5)  = theta_star + theta_bar; % Total theta

x0_sim(3)  = x0(3) - h_bar;

options = odeset('MaxStep', 0.1);
[t_nonlin, x_nonlin_raw] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, [0;0;0], params), [0 250], x0_sim, options);
x_nonlin = x_nonlin_raw'; 

u_nonlin = repmat(u0, 1, length(t_nonlin));

utils.PlotSimulation(t_nonlin, x_nonlin, u_nonlin, 'b'); 



% 
% 
% evals = diag(eval_lon);
% freqs = abs(evals);
% 
% target_freq = 0.629; 
% [~, idx] = min(abs(freqs - target_freq)); 
% 
% phugoid_evec = evec_lon(:, idx);
% 
% target_theta_rad = deg2rad(2);
% current_theta = phugoid_evec(4);
% 
% scaling_factor = target_theta_rad / current_theta;
% x_pert0 = double(real(phugoid_evec * scaling_factor));
% x_pert0 = x_pert0(:);
% 
% 
% 
% % --- 2. Linear Simulation ---
% % Define C as identity to see all states, D as zero
% sys_lon = ss(Alon, zeros(5,1), eye(5), zeros(5,1));
% 
% % Run this BEFORE nonlinear to make sure it works
% [y_lin, t_lin] = initial(sys_lon, x_pert0, 250);
% 
% 
% 
% % --- 3. Nonlinear Simulation ---
% wind_inertial = [0; 0; 0];
% V_trim = 18; 
% alpha_trim = x0(8); % Assuming state 8 is trim alpha in your nonlinear vector
% 
% % Convert alpha perturbation (rad) to w perturbation (m/s)
% w_pert = V_trim * cos(alpha_trim) * x_pert0(2);
% x0_nonlin = real(x0); % Ensure trim is real
% % Mapping: x_pert0 is [u; w; q; theta; h]
% x0_nonlin(7)  = x0(7)  + x_pert0(1); % u = u* + u_bar
% x0_nonlin(9)  = x0(9)  + w_pert;     % w = w* + w_bar
% x0_nonlin(11) = x0(11) + x_pert0(3); % q = q* + q_bar
% x0_nonlin(5)  = x0(5)  + x_pert0(4); % theta = theta* + theta_bar
% x0_nonlin(3)  = x0(3)  - x_pert0(5); % Pd = Pd* - h_bar
% 
% % Use a smaller MaxStep to prevent the "Altitude" warning during large initial steps
% options = odeset('MaxStep', 0.1);
% [t_nonlin, x_nonlin_raw] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, wind_inertial, params), [0 250], x0_nonlin, options);
% x_nonlin = x_nonlin_raw'; 
% 
% % --- 4. Reconstruct Full State for Linear Plotting ---
% x_lin_full = repmat(x0, 1, length(t_lin));
% 
% % NEW: Compute X position by integrating (V_trim + delta_u)
% % cumtrapz performs numerical integration over the time vector
% V_trim = x0(7); 
% x0_nonlin(9)  = x0(9)  + (V_trim * x_pert0(2)); % Convert alpha to w
% delta_u = y_lin(:,1)';
% x_lin_full(1,:) = x0(1) + cumtrapz(t_lin, V_trim + delta_u);
% 
% % Existing mappings
% x_lin_full(7,:)  = x0(7)  + y_lin(:,1)'; % u
% x_lin_full(9,:) = x0(9) + (V_trim * y_lin(:,2)');
% x_lin_full(11,:) = x0(11) + y_lin(:,3)'; % q
% x_lin_full(5,:)  = x0(5)  + y_lin(:,4)'; % theta
% x_lin_full(3,:)  = x0(3)  - y_lin(:,5)'; % Pd = trim_Pd - delta_h
% 
% % --- 5. Plotting ---
% u_nonlin = repmat(u0, 1, length(t_nonlin));
% u_lin = repmat(u0, 1, length(t_lin));
% 
% utils.PlotSimulation(t_nonlin, x_nonlin, u_nonlin, 'b'); 
% utils.PlotSimulation(t_lin, x_lin_full, u_lin, 'r');