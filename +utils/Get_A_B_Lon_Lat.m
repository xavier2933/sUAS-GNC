function [Alon, Blon, Alat, Blat] = Get_A_B_Lon_Lat(trim_def, params)
    Alon = [1];
    Blon = [2];
    Alat = [3];
    Blat = [4];

    [x, u] = utils.CalculateTrim(trim_def, params)
    aircraft_state    = x(:);
    aircraft_surfaces = u(:);
    wind_inertial     = [0;0;0];
    density = utils.stdatmo(-x(3));
    p = params;

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
    V_a = wind_angle_vec(1)
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

    % alpha bs
    C_D_alpha = (2 * p.K * (C_L - p.CLmin)) * p.CLalpha;
    C_X_alpha = - C_D_alpha * cos(alpha) + C_D * sin(alpha) + p.CLalpha * sin(alpha) + C_L * cos(alpha);

    % Xu
    Xu_thrust_term = ((density * p.Sprop * p.Cprop * dt) / p.m) * ((p.kmotor * u_e / V_a) * (1-2*dt) + 2*u_e*(dt-1));
    Xu = (u_e*density*p.S/p.m) * C_X - ((density * p.S * w_e * C_X_alpha) / (2 *p.m)) + Xu_thrust_term;

    % Xw WRONG
    Xw_prop = ((density * p.Sprop * p.Cprop * dt) / p.m) * ((p.kmotor * w_e / V_a) * (1-2*dt) + 2*w_e*(dt-1));
    Xw = ((w_e * density * p.S) / p.m) * C_X + ((density * p.S * C_X_alpha * u_e) / (2 * p.m)) + Xw_prop;

    % Xq
    C_D_q = (2 * p.K * (C_L - p.CLmin)) * p.CLq;
    C_X_q = - C_D_q * cos(alpha) + p.CLq * sin(alpha);
    Xq = -w_e + ((density * V_a * p.S * C_X_q * p.c) / (4 * p.m));

    % Zu WRONG
    C_Z_alpha = - C_D_alpha * sin(alpha) - C_D * cos(alpha) - p.CLalpha * cos(alpha) + C_L * sin(alpha);
    Zu = ((u_e * density * p.S) / p.m) * C_Z - ((density * p.S * C_Z_alpha * w_e) / (2 * p.m));

    % Zw
    Zw = ((w_e * density * p.S) / (p.m)) * C_Z + ((density * p.S * C_Z_alpha * u_e) / (2 * p.m));

    % Zq
    C_Z_q = -C_D_q * sin(alpha) - p.CLq * cos(alpha);
    Zq = u_e + ((density * V_a * p.S * C_Z_q * p.c) /  (4 * p.m));

    Alon = zeros(5,5);
    Alon(1,1) = Xu;
    Alon(1,2) = Xw * V_a;
    Alon(1,3) = Xq;
    Alon(1,4) = - p.g * cos(theta);
    Alon(2,1) = Zu / V_a;
    Alon(2,2) = Zw;
    Alon(2,3) = Zq / V_a;
    Alon(2,4) = - p.g * sin(theta) / V_a;

end