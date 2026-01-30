function [cost] = CostForTrim(trim_variables, trim_definition, params)

   x_dot_trim = zeros(12,1);
   wind = [0;0;0];

   if length(trim_variables) == 3
        [x,u] = utils.TrimConditionFromDefinitionAndVariables(trim_definition, trim_variables);
        xdot = utils.AircraftEOM(0, x, u,wind, params);
        cost = norm(x_dot_trim(4:12) - xdot(4:12));
   else
        x_dot_trim(6) = trim_definition(1) / trim_definition(4);
        [x,u] = utils.TrimConditionFromDefinitionAndVariables(trim_definition, trim_variables);
        xdot = utils.AircraftEOM(0, x, u,wind, params);
        [forces, moments] = utils.AeroForcesAndMoments_BodyState_WindCoeffs(x,u,wind,utils.stdatmo(-x(3)),params);
        cost = norm(x_dot_trim(4:12) - xdot(4:12))^2 + forces(2)^2;
   end


end