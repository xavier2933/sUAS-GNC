clear, clc, close all
addpath('..');

params = utils.ttwistor();

% test from hw4
% vars = [20;0;200];

vars = [18;0;1800];

[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params)

% For Longitudinal Modes
fprintf('Longitudinal Modes:\n');
damp(Alon)

% For Lateral Modes
fprintf('\nLateral Modes:\n');
damp(Alat)

[evec_lat, eval_lat] = eig(Alat)


% res = utils.RotationMatrix321([0, pi/3, pi/2])
% res'
% 
% v_e_b = [10;-2;3];
% v_a_b = [10;-1;-2];
% 
% res = v_e_b - v_a_b
% 
% res_e = utils.TransformFromBodyToInertial(res,[0, pi/3, pi/2])

