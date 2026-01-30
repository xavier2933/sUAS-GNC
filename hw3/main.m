clear, clc, close all
addpath('..');


% [trim,cs] = utils.TrimConditionFromDefinitionAndVariables([0;10;-1000],[0;0.2;0])
% 
% ap = utils.ttwistor();
% 
% [x, u] = utils.CalculateTrim([30;0.2;-1000], ap);
% [x1, u1] = utils.CalculateTrim([30;0.2;1000;1000], ap);

params = utils.ttwistor();

state = [1000, 1000, 1000, 0.0, 0.0, 0.0, 1.0, 1.2, 1.2, 0.1, 0.1, 0.1];
params = utils.ttwistor();
surfaces = [0.0, 0.0, 0.0, 0.0];
wind = [0,0,0];

% Initial time span
t0 = 0;
tf = 300;          % seconds
tspan = [t0 tf];


% q1 good
% vars = [18;0;1655];
% % 
% [x0,u0] = utils.CalculateTrim(vars, params);
% [t, x] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, wind, params), tspan, x0);
% ctrl = repmat(u0, 1, length(t));
% 
% utils.PlotSimulation(t,x',ctrl,'b')
% 
% q2
% x1 = x0
% w_b = utils.TransformFromInertialToBody([10;10;0], x1(4:6))
% x1(7:9) = x1(7:9) + w_b;
% 
% [t, x] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, [10;10;0], params), tspan, x1);
% ctrl = repmat(u0, 1, length(t));
% 
% utils.PlotSimulation(t,x',ctrl,'r')
% 
% q3
% vars = [18;deg2rad(10);1655];
% [x2,u2] = utils.CalculateTrim(vars, params)
% [t, x] = ode45(@(t, x) utils.AircraftEOM(t, x, u2, wind, params), tspan, x2);
% ctrl = repmat(u2, 1, length(t));
% 
% utils.PlotSimulation(t,x',ctrl,'c')

% q4 - check the trim vals
vars = [19.9;0;200;500];

[x3,u3] = utils.CalculateTrim(vars, params)
[t, x] = ode45(@(t, x) utils.AircraftEOM(t, x, u3, [0;0;0], params), tspan, x3);
ctrl = repmat(u3, 1, length(t));

utils.PlotSimulation(t,x',ctrl,'r')

