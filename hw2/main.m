clear, clc, close all
addpath('..');

aircraft_state = [1;2;3;4;5;6;7;8;9;10;11;12];
control_surfaces = [1;2;3;4];
wind_inertial = [1;2;3]
density =1.1
params = utils.ttwistor();


[aero_f, aero_m] = utils.AeroForcesAndMoments_BodyState_WindCoeffs(aircraft_state, control_surfaces, wind_inertial, density, params)
[aircraft_f, aircraft_m] = utils.AircraftForcesAndMoments(aircraft_state, control_surfaces, wind_inertial, density, params)
xdot = utils.AircraftEOM(0, aircraft_state,control_surfaces,wind_inertial, params)



state = [1000, 1000, 1000, 0.0, 0.0, 0.0, 1.0, 1.2, 1.2, 0.1, 0.1, 0.1];
params = utils.ttwistor();
surfaces = [0.0, 0.0, 0.0, 0.0];
wind = [0,0,0];
wind2 = [10,10,0];

% 
% Initial time span
t0 = 0;
tf = 300;          % seconds
tspan = [t0 tf];
% 
% % Initial state (12x1)
v_b = utils.WindAnglesToAirRelativeVelocityVector([18;0;0])

x0 = zeros(12, 1);
x0(3) = -1655;
x0(7) = v_b(1);
x0(8) = v_b(2);
x0(9) = v_b(3);
x0
% 
% %[forces, moments] = utils.AircraftForcesAndMoments(x0, surfaces, wind, utils.stdatmo(x0(3)), params)
% 
% q1
[t, x] = ode45(@(t, x) utils.AircraftEOM(t, x, surfaces, wind, params), tspan, x0);
ctrl = zeros(4, length(t));

utils.PlotSimulation(t,x',ctrl,'b')

% q2
[t2, x2] = ode45(@(t, x) utils.AircraftEOM(t, x, surfaces, wind2, params), tspan, x0);
ctrl = zeros(4, length(t2));

utils.PlotSimulation(t2,x2',ctrl,'r')
% 
% % q3
% x03 = [0;0;-1800;deg2rad(15);deg2rad(-12);deg2rad(270);19;3;-2;deg2rad(0.08);deg2rad(-0.2);deg2rad(0)];
% u0 = [deg2rad(5);deg2rad(2);deg2rad(-13);deg2rad(0.3)];
% 
% [t3, x3] = ode45(@(t, x) utils.AircraftEOM(t, x, u0, wind, params), tspan, x03);
% ctrl = zeros(4, length(t3));
% 
% utils.PlotSimulation(t3,x3',ctrl,'c')