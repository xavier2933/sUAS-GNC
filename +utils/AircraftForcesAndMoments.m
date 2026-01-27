function [aircraft_forces, aircraft_moments] = AircraftForcesAndMoments(aircraft_state, aircraft_surfaces, wind_inertial, density, aircraft_parameters)
    aircraft_state    = aircraft_state(:);
    aircraft_surfaces = aircraft_surfaces(:);
    wind_inertial     = wind_inertial(:);
    [aero_force, aero_moment] = utils.AeroForcesAndMoments_BodyState_WindCoeffs(aircraft_state, aircraft_surfaces, wind_inertial, density, aircraft_parameters);
    
    phi   = aircraft_state(4);
    theta = aircraft_state(5);
    psi   = aircraft_state(6);

    angles = [phi;theta;psi];

    R_e_b = utils.RotationMatrix321(angles);
    F_g = R_e_b * [0;0;aircraft_parameters.m * 9.81];

    F_g_b = aero_force + F_g;
    
    aircraft_forces = F_g_b; 
    aircraft_moments = aero_moment;
end
