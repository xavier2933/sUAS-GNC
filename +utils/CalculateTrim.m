function [aircraft_state_trim, control_surface_trim, fval] = CalculateTrim(trim_definition, params)

    if length(trim_definition) == 3
        x0 = [0;0;0.2];
        lb = [-45*pi/180; -45*pi/180; 0];
        ub = [ 45*pi/180;  45*pi/180; 1];
    else
        x0 = [2*pi/180; 0; 0.5; 0.08; 0; 0; 0];
        lb = [-2*pi; -2*pi; 0; -45*pi/180; -45*pi/180;-45*pi/180; -45*pi/180];
        ub = [ 2*pi;  2*pi;  1;  45*pi/180; 45*pi/180; 45*pi/180; 45*pi/180;];
    end

    cost_fun = @(x) utils.CostForTrim(x, trim_definition, params);

    options = optimoptions('fmincon', 'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-8);

    [optimal_vars, fval] = fmincon(cost_fun, x0, [], [], [], [], lb, ub, [], options)

    [aircraft_state_trim, control_surface_trim] = utils.TrimConditionFromDefinitionAndVariables(trim_definition, optimal_vars);
end