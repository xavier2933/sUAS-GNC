function [vector_body] = TransformFromInertialToBody(vector_inertial, euler_angles)
% For a vector given in inertial coordinates, determine the components in body coordinates
vector_inertial = vector_inertial(:);
euler_angles = euler_angles(:);
R = utils.RotationMatrix321(euler_angles);
vector_body = R * vector_inertial;

end