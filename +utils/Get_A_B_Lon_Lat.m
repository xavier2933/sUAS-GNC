function [Alon, Blon, Alat, Blat] = Get_A_B_Lon_Lat(trim_def, params)
    Alon = [1];
    Blon = [2];
    Alat = [3];
    Blat = [4];

    [x, u] = utils.CalculateTrim(trim_def, params);
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

    % alpha bs
    C_D_alpha = (2 * p.K * (C_L - p.CLmin)) * p.CLalpha;
    C_X_alpha = - C_D_alpha * cos(alpha) + C_D * sin(alpha) + p.CLalpha * sin(alpha) + C_L * cos(alpha);

    % Xu
    Xu_thrust_term = ((density * p.Sprop * p.Cprop * dt) / p.m) * ((p.kmotor * u_e / V_a) * (1-2*dt) + 2*u_e*(dt-1));
    Xu = (u_e*density*p.S/p.m) * C_X - ((density * p.S * w_e * C_X_alpha) / (2 *p.m)) + Xu_thrust_term;

    % Xw
    Xw_prop = ((density * p.Sprop * p.Cprop * dt) / p.m) * ((p.kmotor * w_e / V_a) * (1-2*dt) + 2*w_e*(dt-1));
    Xw = ((w_e * density * p.S) / p.m) * C_X + ((density * p.S * C_X_alpha * u_e) / (2 * p.m)) + Xw_prop;

    % Xq
    C_D_q = (2 * p.K * (C_L - p.CLmin)) * p.CLq;
    C_X_q = - C_D_q * cos(alpha) + p.CLq * sin(alpha);
    Xq = -w_e + ((density * V_a * p.S * C_X_q * p.c) / (4 * p.m));

    % Xde CHECK
    % CXde = p.CLde * (sin(alpha) - (2 * p.K * (C_L - p.CLmin)*cos(alpha)));
    CXde = -cos(alpha)*(2 * p.K * (C_L - p.CLmin) * p.CLde) + sin(alpha) * p.CLde;
    Xde = density * V_a^2 * p.S * CXde / (2 * p.m);

    % Xdt
    Xdt = (density * p.Sprop * p.Cprop / p.m) * (V_a * (p.kmotor - V_a) + 2 * dt * (p.kmotor - V_a)^2);

    % Zu
    C_Z_alpha = - C_D_alpha * sin(alpha) - C_D * cos(alpha) - p.CLalpha * cos(alpha) + C_L * sin(alpha);
    Zu = ((u_e * density * p.S) / p.m) * C_Z - ((density * p.S * C_Z_alpha * w_e) / (2 * p.m));

    % Zw
    Zw = ((w_e * density * p.S) / (p.m)) * C_Z + ((density * p.S * C_Z_alpha * u_e) / (2 * p.m));

    % Zq
    C_Z_q = -C_D_q * sin(alpha) - p.CLq * cos(alpha);
    Zq = u_e + ((density * V_a * p.S * C_Z_q * p.c) /  (4 * p.m));

    % Zde
    CZde = -sin(alpha)*(2 * p.K * (C_L - p.CLmin) * p.CLde) - cos(alpha) * p.CLde;
    Zde = (density * V_a^2 * p.S * CZde) / (2 * p.m);

    % Mu
    Mu = (u_e * density * p.S * p.c/p.Iy)*(p.Cm0 + p.Cmalpha * alpha + p.Cmde * de);
    Mu = Mu - (density * p.S * p.c * p.Cmalpha * w_e/(2*p.Iy));

    % Mw
    Mw = (w_e * density * p.S * p.c/p.Iy)*(p.Cm0 + p.Cmalpha * alpha + p.Cmde * de);
    Mw = Mw + (density * p.S * p.c * p.Cmalpha * u_e/(2*p.Iy));

    % Mq
    Mq = density * V_a * p.S * p.c^2 * p.Cmq / (4 * p.Iy);

    % Mde
    Mde = (density * V_a^2 * p.S * p.c * p.Cmde) / (2 * p.Iy);

    % LATERAL
    % Yv
    Yv = (density * p.S * v_e / p.m) * (p.CY0 + p.CYbeta * beta + p.CYda * da + p.CYdr * dr);
    Yv = Yv + (density * p.S * p.CYbeta / (2 * p.m)) * (u_e^2 + w_e^2)^(1/2);

    % Yp
    Yp = w_e + (density * V_a * p.S * p.b / ( 4 * p.m)) * p.CYp;

    % Yr
    Yr = -u_e + (density * V_a * p.S * p.b / (4 * p.m)) * p.CYr;

    % Yda 
    Yda = (density * V_a^2 * p.S / (2 * p.m)) * p.CYda;

    % Ydr
    Ydr = (density * V_a^2 * p.S / (2 * p.m)) * p.CYdr;

    % Lv
    Lv = (density * p.S * p.b * v_e) * (p.Cp0 + p.Cpbeta * beta + p.Cpda * da + p.Cpdr * dr);
    Lv = Lv + (density * p.S * p.b * p.Cpbeta / (2)) * (u_e^2 + w_e^2)^(1/2);

    % Lp
    gamma = p.Ix * p.Iz - p.Ixz^2;
    gamma1 = (p.Ixz * (p.Ix - p.Iy + p.Iz)) / gamma;
    Lp = (density * V_a *p.S * p.b^2 /4) * p.Cpp;

    % Lr
    gamma2 = (p.Iz * (p.Iz - p.Iy) + p.Ixz^2) / gamma;
    Lr = (density * V_a * p.S * p.b^2 / 4) * p.Cpr;

    % Lda
    Lda = (density * V_a^2 * p.S * p.b / 2) * p.Cpda;

    % Ldr
    Ldr = (density * V_a^2 * p.S * p.b / 2) * p.Cpdr; 

    % Nv
    Nv = (density * p.S * p.b * v_e) * (p.Cr0 + p.Crbeta * beta + p.Crda * da + p.Crdr * dr);
    Nv = Nv + (density * p.S * p.b * p.Crbeta / (2)) * (u_e^2 + w_e^2)^(1/2);

    % Np
    Np = (density * V_a * p.S * p.b^2 / 4) * p.Crp;

    % Nr
    Nr = (density * V_a * p.S * p.b^2 / 4) * p.Crr;

    % Nda
    Nda = (density * V_a^2 * p.S * p.b / 2) * p.Crda; 

    % Ndr
    Ndr = (density * V_a^2 * p.S * p.b / 2) * p.Crdr; 



    Alon = zeros(5,5);
    Alon(1,1) = Xu;
    Alon(1,2) = Xw * V_a * cos(alpha);
    Alon(1,3) = Xq;
    Alon(1,4) = - p.g * cos(theta);
    Alon(2,1) = Zu / (V_a * cos(alpha));
    Alon(2,2) = Zw;
    Alon(2,3) = Zq / (V_a * cos(alpha));
    Alon(2,4) = - p.g * sin(theta) / (V_a * cos(alpha));
    Alon(3,1) = Mu;
    Alon(3,2) = Mw * V_a * cos(alpha);
    Alon(3,3) = Mq;
    Alon(4,3) = 1;
    Alon(5,1) = sin(theta);
    Alon(5,2) = -V_a * cos(theta) * cos(alpha);
    Alon(5,4) = u_e * cos(theta) + w_e * sin(theta);

    Blon = zeros(5,2);
    Blon(1,1) = Xde;
    Blon(1,2) = Xdt;
    Blon(2,1) = Zde / (V_a * cos(alpha));
    Blon(3,1) = Mde;

    denom = V_a * cos(beta);
    Alat = zeros(5,5);
    Alat(1,1) = Yv;
    Alat(1,2) = Yp/denom;
    Alat(1,3) = Yr/denom;
    Alat(1,4) = (p.g*cos(theta)*(cos(phi))) / denom;
    Alat(2,1) = Lv * V_a * cos(beta);
    Alat(2,2) = Lp;
    Alat(2,3) = Lr;
    Alat(3,1) = Nv * V_a * cos(beta);
    Alat(3,2) = Np;
    Alat(3,3) = Nr;
    Alat(4,2) = 1;
    Alat(4,3) = cos(phi) * tan(theta);
    Alat(5,3) = cos(phi) * sec(theta);


    Blat = zeros(5,2);
    Blat(1,1) = Yda / denom;
    Blat(1,2) = Ydr / denom;
    Blat(2,1) = Lda;
    Blat(2,2) = Ldr;
    Blat(3,1) = Nda;
    Blat(3,2) = Ndr;

end