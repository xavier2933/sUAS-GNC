function [C_X, C_Z, C_L] = Calc_CX_CZ(aircraft_state, aircraft_surfaces, wind_inertial, density, aircraft_parameters)
    aircraft_state    = aircraft_state(:);
    aircraft_surfaces = aircraft_surfaces(:);
    wind_inertial     = wind_inertial(:);
    p = aircraft_parameters;

    % extract state
    x_e = aircraft_state(1);
    y_e = aircraft_state(2);
    z_e = aircraft_state(3);
    phi = aircraft_state(4);
    theta = aircraft_state(5);
    psi = aircraft_state(6);
    u_e = aircraft_state(7);
    v_e = aircraft_state(8);
    w_e = aircraft_state(9);
    p_s = aircraft_state(10);
    q = aircraft_state(11);
    r = aircraft_state(12);

    % extract control surfaces
    de = aircraft_surfaces(1);
    da = aircraft_surfaces(2);
    dr = aircraft_surfaces(3);
    dt = aircraft_surfaces(4);

    % get wind angles
    V_e_b = [u_e; v_e; w_e];
    angles = [phi, theta, psi];
    wind_body = utils.TransformFromInertialToBody(wind_inertial', angles);
    V_a_b = V_e_b - wind_body;
    %V_a_b = utils.TransformFromInertialToBody(V_a_e', angles);
    wind_angle_vec = utils.AirRelativeVelocityVectorToWindAngles(V_a_b);
    V_a = wind_angle_vec(1);
    beta = wind_angle_vec(2);
    alpha = wind_angle_vec(3);

    % get nondimensional rates
    p_hat = (p_s * p.b) / (2 * V_a);
    q_hat = (q * p.c) / (2 * V_a);
    r_hat = (r * p.b) / (2 * V_a);

    % Nondimensional coefs
    C_L = p.CL0 + p.CLalpha * alpha + p.CLq * q_hat + p.CLde * de;
    C_D = p.CDmin + p.K * (C_L - p.CLmin)^2;
    C_T = ((2 * p.Sprop * p.Cprop * dt) / (p.S * V_a^2)) * (V_a + dt * (p.kmotor - V_a)) * (p.kmotor - V_a);
    C_Y = p.CYbeta * beta + p.CYp * p_hat + p.CYr * r_hat + p.CYda * da + p.CYdr * dr;
    C_l = p.Clbeta * beta + p.Clp * p_hat + p.Clr * r_hat + p.Clda * da + p.Cldr * dr;
    C_m = p.Cm0 + p.Cmalpha * alpha + p.Cmq * q_hat + p.Cmde * de;
    C_n = p.Cnbeta * beta + p.Cnp * p_hat + p.Cnr * r_hat + p.Cnda * da + p.Cndr * dr;

    % body x and z coefs
    C_X = -C_D * cos(alpha) + C_L * sin(alpha);
    C_Z = - C_D * sin(alpha) - C_L * cos(alpha);
end


