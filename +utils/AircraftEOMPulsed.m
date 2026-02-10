function [xdot] = AircraftEOMPulsed(time, aircraft_state, aircraft_surfaces, wind_inertial, aircraft_parameters)
    
    if time < 1
        del_pulse = 1;
    else
        del_pulse = 0;
    end

    aircraft_state    = aircraft_state(:);
    aircraft_surfaces = aircraft_surfaces(:);
    aircraft_surfaces(1) = aircraft_surfaces(1) + del_pulse;
    wind_inertial     = wind_inertial(:);
    angles = [aircraft_state(4); aircraft_state(5); aircraft_state(6)];
    V_e_b = [aircraft_state(7); aircraft_state(8); aircraft_state(9)];
    omega_b = [aircraft_state(10); aircraft_state(11); aircraft_state(12)];

    phi = aircraft_state(4); 
    theta = aircraft_state(5); 
    psi = aircraft_state(6);

    p_dot = utils.TransformFromBodyToInertial(V_e_b, angles);

    T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
        0  cos(phi)            -sin(phi);
        0  sin(phi)*sec(theta) cos(phi)*sec(theta)];

    o_dot = T * omega_b;

    density = utils.stdatmo(-aircraft_state(3));

    [aero_force, aero_moment] = utils.AircraftForcesAndMoments(aircraft_state, aircraft_surfaces, wind_inertial, density, aircraft_parameters);
    
    v_dot_e_b = cross(-omega_b, V_e_b) + aero_force / aircraft_parameters.m;

    I = aircraft_parameters.inertia_matrix;

    w_dot_b = I \ (cross(-omega_b, I*omega_b) + aero_moment);

    xdot = [p_dot;o_dot; v_dot_e_b;w_dot_b];
end