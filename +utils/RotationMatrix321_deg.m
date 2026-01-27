function [R] = RotationMatrix321_deg(euler_angles)
% Computes 321 matrix with angles given in [deg]

phi = euler_angles(1);
theta = euler_angles(2);
psi = euler_angles(3);

R3 = [cosd(phi) sind(phi) 0; -sind(phi) cosd(phi) 0; 0 0 1];
R2 = [cosd(theta) 0 -sind(theta); 0 1 0; sind(theta) 0 cosd(theta)];
R1 = [1 0 0; 0 cosd(psi) sind(psi); 0 -sind(psi) cosd(psi)];

R = R1 * R2 * R3;
end