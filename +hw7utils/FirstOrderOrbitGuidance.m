function vel = FirstOrderOrbitGuidance(t, pos, orbit_speed, orbit_radius, orbit_center, orbit_flag, orbit_gains)
    pn = pos(1); pe = pos(2); pd = pos(3);
    
    d = sqrt((pn - orbit_center(1))^2 + (pe - orbit_center(2))^2);
    
    phi = atan2(pe - orbit_center(2), pn - orbit_center(1));
    chi_c = phi + orbit_flag * (pi/2 + atan(orbit_gains.kr * (d - orbit_radius) / orbit_radius));
    
    altitude_error = orbit_center(3) - pd; 
    v_d = orbit_gains.kz * altitude_error;
    
    v_n = orbit_speed * cos(chi_c);
    v_e = orbit_speed * sin(chi_c);
    
    vel = [v_n; v_e; v_d]; 
end