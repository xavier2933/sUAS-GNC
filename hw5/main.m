clear, clc, close all
addpath('..');


vars = [18;0;1800];
params = utils.ttwistor();

[x0,u0] = utils.CalculateTrim(vars, params)
[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params)

% For Longitudinal Modes
fprintf('Longitudinal Modes:\n');
damp(Alon)
[evec_lon, eval_lon] = eig(Alon);


% For Lateral Modes
fprintf('\nLateral Modes:\n');
damp(Alat);

[evec_lat, eval_lat] = eig(Alat);



evals = diag(eval_lon);
freqs = abs(evals);

target_freq = 0.629; 
[~, idx] = min(abs(freqs - target_freq)); 

phugoid_evec = evec_lon(:, idx);

target_theta_rad = deg2rad(2);
current_theta = phugoid_evec(4);

scaling_factor = target_theta_rad / current_theta;
x_pert0 = double(real(phugoid_evec * scaling_factor));
x_pert0 = x_pert0(:);

% --- 2. Linear Simulation ---
% Define C as identity to see all states, D as zero
sys_lon = ss(Alon, zeros(5,1), eye(5), zeros(5,1));

% Run this BEFORE nonlinear to make sure it works
[y_lin, t_lin] = initial(sys_lon, x_pert0, 250); 

% --- 3. Nonlinear Simulation ---
wind_inertial = [0; 0; 0];
x0_nonlin = real(x0); % Ensure trim is real
% Mapping: x_pert0 is [u; w; q; theta; h]
x0_nonlin(7)  = x0(7)  + x_pert0(1); % u
x0_nonlin(9)  = x0(9)  + x_pert0(2); % w
x0_nonlin(11) = x0(11) + x_pert0(3); % q
x0_nonlin(5)  = x0(5)  + x_pert0(4); % theta
x0_nonlin(3)  = x0(3)  - x_pert0(5); % Pd = -h

% Use a smaller MaxStep to prevent the "Altitude" warning during large initial steps
options = odeset('MaxStep', 0.1);
[t_nonlin, x_nonlin_raw] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, wind_inertial, params), [0 250], x0_nonlin, options);
x_nonlin = x_nonlin_raw'; 

% --- 4. Reconstruct Full State for Linear Plotting ---
x_lin_full = repmat(x0, 1, length(t_lin));

% NEW: Compute X position by integrating (V_trim + delta_u)
% cumtrapz performs numerical integration over the time vector
V_trim = x0(7); 
delta_u = y_lin(:,1)';
x_lin_full(1,:) = x0(1) + cumtrapz(t_lin, V_trim + delta_u);

% Existing mappings
x_lin_full(7,:)  = x0(7)  + y_lin(:,1)'; % u
x_lin_full(9,:)  = x0(9)  + y_lin(:,2)'; % w
x_lin_full(11,:) = x0(11) + y_lin(:,3)'; % q
x_lin_full(5,:)  = x0(5)  + y_lin(:,4)'; % theta
x_lin_full(3,:)  = x0(3)  - y_lin(:,5)'; % Pd = trim_Pd - delta_h

% --- 5. Plotting ---
u_nonlin = repmat(u0, 1, length(t_nonlin));
u_lin = repmat(u0, 1, length(t_lin));

utils.PlotSimulation(t_nonlin, x_nonlin, u_nonlin, 'b'); 
utils.PlotSimulation(t_lin, x_lin_full, u_lin, 'r');