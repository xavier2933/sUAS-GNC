function xdot = KinematicGuidanceModel(t, x, wind, control_objective, ap)
% x = [pn pe chi chidot h hdot Va
% control_objectives = [h_c, hdot_c, chi_c, chidot_c, Va_c
    bc = ap.bc;
    bcd = ap.bcd;
    bh = ap.bh;
    bhd = ap.bhd;
    bva = ap.bva;
    Va = x(7);
    chi = x(3);
    chidot_c = control_objective(4);
    chidot = x(4);
    chi_c = control_objective(3);
    h = x(5);
    h_c = control_objective(1);
    Va_c = control_objective(5);
    hdot_c = control_objective(2);

    psi = chi - asin((1/Va) * [wind(1,1), wind(2,1)] * [-sin(chi);cos(chi)]);
    pnd = Va * cos(psi) + wind(1,1);
    ped = Va * sin(psi) + wind(2,1);
    chi_dd = bcd * (chidot_c - chidot) + bc * (chi_c - chi);
    hd = x(6);
    h_dd = bhd * (hdot_c - hd) + bh * (h_c - h);
    Va_d = bva * (Va_c - Va);
    chi_d = chidot;

    xdot = [[pnd;ped];chi_d;chi_dd;hd;h_dd;Va_d];
end