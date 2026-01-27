clear, clc, close all;
addpath('..')
% 
% 
% rot = utils.RotationMatrix321_deg([9,-2,33])
% state = utils.ttwistor()
% density = utils.stdatmo(5280)

phi = 9
theta = -2
psi = 33

R3 = [cosd(phi) sind(phi) 0; -sind(phi) cosd(phi) 0; 0 0 1]
R2 = [cosd(theta) 0 -sind(theta); 0 1 0; sind(theta) 0 cosd(theta)]
R1 = [1 0 0; 0 cosd(psi) sind(psi); 0 -sind(psi) cosd(psi)]

R = R1 * R2 * R3

r = [0.8382 0.5443 0.0349; -0.5425 0.8254 0.1563; 0.0563 -0.15 0.9871];


res = r * [18; 0; -5]