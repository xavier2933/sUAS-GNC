function vel = FirstOrderStraightLineGuidance(t, pos, pos_line, dir_line, kpath, chi_inf, V_trim)
    
    control_objectives = utils.StraightLineGuidance(pos_line, dir_line, pos, kpath, chi_inf, V_trim);
    kh = 0.1;
    h_c = control_objectives(1);
    h_dot_c = control_objectives(2);
    z_vel = -(h_dot_c + kh * (h_c + pos(3)));

    chi_c = control_objectives(3);

    if (z_vel^2 > control_objectives(5)^2)
        fprintf(1, 'Climb rate too high \n')
        z_vel = -h_dot_c;
    end
    plane_speed = sqrt(control_objectives(5)^2 - z_vel^2);
    vel = [plane_speed*cos(chi_c);plane_speed*sin(chi_c); z_vel];
end
