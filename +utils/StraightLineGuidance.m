function control_objectives = StraightLineGuidance(pos_line, dir_line, pos, kpath, chi_inf, V_trim)
    q_n = dir_line(1);
    q_e = dir_line(2);
    p_n = pos(1);
    p_e = pos(2);
    r_n = pos_line(1);
    r_e = pos_line(2);
    chi_q = atan2(q_e, q_n);
    chi = atan2(p_e, p_n);
    while chi_q - chi < -pi
        chi_q = chi_q + 2*pi;
    end
    while chi_q - chi > pi
        chi_q = chi_q - 2*pi;
    end

    e_py = -sin(chi_q)*(p_n - r_n) + cos(chi_q)*(p_e - r_e);
    chi_c = chi_q - chi_inf * (2/pi) * atan(kpath * e_py);
    control_objectives = [-pos_line(3);0;chi_c;0;V_trim];
end