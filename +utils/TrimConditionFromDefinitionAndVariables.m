function [x_trim, control_vec] = TrimConditionFromDefinitionAndVariables(trim_definition, trim_variables)
   trim_definition = trim_definition(:);
   trim_variables = trim_variables(:);


   if length(trim_definition) == 3
       Va = trim_definition(1);
       gamma_0 = trim_definition(2);
       h_0 = trim_definition(3);

       alpha_zero = trim_variables(1);
       delta_e_0 = trim_variables(2);
       delta_t_0 = trim_variables(3);

       beta = 0;

       x_star = zeros(12,1);
       x_star(3) = -h_0;
       x_star(5) = gamma_0 + alpha_zero;
       v_b = utils.WindAnglesToAirRelativeVelocityVector([Va;beta;alpha_zero]);
       x_star(7) = v_b(1);
       x_star(9) = v_b(3);

       x_trim = x_star;

       u = zeros(4,1);
       u(1) = delta_e_0;
       u(4) = delta_t_0;
       control_vec = u;
   else
       Va = trim_definition(1);
       gamma_0 = trim_definition(2);
       h_0 = trim_definition(3);
       R_0 = trim_definition(4);

       alpha_zero = trim_variables(1);
       delta_e_0 = trim_variables(2);
       delta_t_0 = trim_variables(3);
       phi_0 = trim_variables(4);
       beta_0 = trim_variables(5);
       delta_a_0 = trim_variables(6);
       delta_r_0 = trim_variables(7);

       x_star = zeros(12,1);
       x_star(3) = -h_0;
       x_star(4) = phi_0;
       theta = gamma_0 + alpha_zero;
       x_star(5) = theta;
       v_b = utils.WindAnglesToAirRelativeVelocityVector([Va;beta_0;alpha_zero]);
       x_star(7) = v_b(1);
       x_star(8) = v_b(2);
       x_star(9) = v_b(3);

       chi = (Va / R_0)* cos(gamma_0);
       x_star(10) = -chi * sin(theta);
       x_star(11) = chi * sin(phi_0) * cos(theta);
       x_star(12) = chi * cos(phi_0) * cos(theta);

       x_trim = x_star;

       u = zeros(4,1);
       u(1) = delta_e_0;
       u(2) = delta_a_0;
       u(3) = delta_r_0;
       u(4) = delta_t_0;
       control_vec = u;
   end

end