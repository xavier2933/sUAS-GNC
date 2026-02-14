clear, clc, close all
addpath('..');

vars = [18;0;1800];
params = utils.ttwistor();
time = 150; % seconds
control_idx = 2;
pulse_val = 10;
doublet = true;

[x0,u0] = utils.CalculateTrim(vars, params)

options = odeset('MaxStep', 0.1);
[t, x] = ode45(@(t, x) utils.AircraftEOMPulsed(t, x, u0, [0;0;0], params, control_idx, pulse_val, doublet), [0 time], x0, options);


u = repmat(u0, 1, length(t));
u(control_idx, t < 1) = u(control_idx, t < 1) + pulse_val;

utils.PlotSimulation(t, x', u, 'b');


control_idx = 3;
pulse_val = 20;


[x0,u0] = utils.CalculateTrim(vars, params)

options = odeset('MaxStep', 0.1);
[t, x] = ode45(@(t, x) utils.AircraftEOMPulsed(t, x, u0, [0;0;0], params, control_idx, pulse_val, doublet), [0 time], x0, options);


u = repmat(u0, 1, length(t));
u(control_idx, t < 1) = u(control_idx, t < 1) + pulse_val;

utils.PlotSimulation(t, x', u, 'r');