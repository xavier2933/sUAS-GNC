function g = OrbitGuidance(pos, orbit_speed, orbit_radius, orbit_center, orbit_flag, orbit_gains)
    pn = pos(1);
    pe = pos(2);
    
    d = sqrt((pn - orbit_center(1))^2 + (pe - orbit_center(2))^2);
    phi = atan2(pe - orbit_center(2), pn - orbit_center(1));
    
    chi_g = phi + orbit_flag * (pi/2 + atan(orbit_gains.kr * (d - orbit_radius) / orbit_radius));
    
    dt = 0.1; % match line 104 RunHW7
    v_n = orbit_speed * cos(chi_g);
    v_e = orbit_speed * sin(chi_g);
    
    pn_plus = pn + v_n * dt;
    pe_plus = pe + v_e * dt;
    
    d_plus = sqrt((pn_plus - orbit_center(1))^2 + (pe_plus - orbit_center(2))^2);
    phi_plus = atan2(pe_plus - orbit_center(2), pn_plus - orbit_center(1));
    chi_g_plus = phi_plus + orbit_flag * (pi/2 + atan(orbit_gains.kr * (d_plus - orbit_radius) / orbit_radius));
    
    chi_dot_ff = (chi_g_plus - chi_g) / dt; 
    
    h_c = -orbit_center(3);

    g = [h_c; 0; chi_g; chi_dot_ff; orbit_speed];
end