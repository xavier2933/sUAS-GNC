function [R] = RotationMatrix321(euler_angles)
% [phi, theta, psi]
% Computes 321 matrix with angles given in [RADS]

phi = euler_angles(1);
theta = euler_angles(2);
psi = euler_angles(3);

R3 = [cos(psi)  sin(psi) 0; 
     -sin(psi)  cos(psi) 0; 
      0         0        1];

R2 = [cos(theta) 0 -sin(theta); 
      0          1  0; 
      sin(theta) 0  cos(theta)];

R1 = [1  0        0; 
      0  cos(phi) sin(phi); 
      0 -sin(phi) cos(phi)];

R = R1 * R2 * R3; 
end
