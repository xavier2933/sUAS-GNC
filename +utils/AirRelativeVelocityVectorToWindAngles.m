function [wind_angles] = AirRelativeVelocityVectorToWindAngles(velocity_body)
%Given the air relative velocity vector in body coordinates vB, this function returns the
% wind angles in the column vector [Va, β, α]'
u_r = velocity_body(1);
v_r = velocity_body(2);
w_r = velocity_body(3);

Va = sqrt(u_r^2 + v_r^2 + w_r^2);
beta = asin(v_r / Va);
alpha = atan2(w_r, u_r);

wind_angles = [Va; beta; alpha];

end