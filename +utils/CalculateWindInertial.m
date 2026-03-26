function [V_a, V_a_e, w_e_e] = CalculateWindInertial(v_tip, euler_angles, angular_rates, wind_angles)
% v_tip = nx3 inertial velocities inertial frame
% euler_angles = nx3 euler angles (rad)
% angular_rates = angular rates, rad/s
% wind_angles = [airspeed, beta, alpha]
% n is number of measurements
    V_a = utils.WindAnglesToAirRelativeVelocityVector(wind_angles); % body coords
    V_a_e = utils.TransformFromBodyToInertial(V_a, euler_angles);
    w_e_e = V_a_e - v_tip;
    
end