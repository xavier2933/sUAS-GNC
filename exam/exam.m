clear, clc, close all
addpath('..');

params = utils.raaven();

vars = [21;0;1500];

[x0,u0] = utils.CalculateTrim(vars, params)

[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params)

% For Longitudinal Modes
fprintf('Longitudinal Modes:\n');
damp(Alon)

% For Lateral Modes
fprintf('\nLateral Modes:\n');
damp(Alat)



[V,D] = eig(Alat);

lambda = diag(D);

idx = find(lambda > 0);   % spiral mode
lambda_s = lambda(idx)

v = V(:,idx)


vr = v(3);
vphi = v(4);

ratio = vr/vphi
ratio * 0.0524