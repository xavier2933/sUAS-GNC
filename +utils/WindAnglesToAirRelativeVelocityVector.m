function [velocity_body] = WindAnglesToAirRelativeVelocityVector(wind_angles)
% Calculate the aircraft air relative velocity vector in body coordinates from the airspeed,
% sideslip, and angle of attack, (the wind angles). The input and output of the function should
% be three-dimensional column vectors.
% input: [V_a, beta, alpha] RADIANS
Va = wind_angles(1);
beta = wind_angles(2);
alpha = wind_angles(3);

velocity_body = Va * [cos(alpha) * cos(beta); sin(beta); sin(alpha) * cos(beta)];
end