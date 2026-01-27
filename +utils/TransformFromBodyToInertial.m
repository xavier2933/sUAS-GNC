function [vector_inertial] = TransformFromBodyToInertial(vector_body, euler_angles)
% For a vector given in body coordinates, determine the components in inertial coordinates.
vector_body = vector_body(:);
euler_angles = euler_angles(:);
R = utils.RotationMatrix321(euler_angles);
vector_inertial = R' * vector_body;

end