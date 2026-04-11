function [sVw, sChi, sWz] = get_sensitivities(v_tip, euler, wind_angs, div)
    [~, ~, we_nom] = utils.CalculateWindInertial(v_tip, euler, 0, wind_angs);
    Vw_nom  = sqrt(we_nom(1)^2 + we_nom(2)^2);
    chi_nom = atan2(we_nom(2), we_nom(1));
    wz_nom  = we_nom(3);

    sVw = zeros(1,6); sChi = zeros(1,6); sWz = zeros(1,6);

    for i = 1:6
        e_pert = euler; 
        w_pert = wind_angs;
        
        if i <= 3
            e_pert(i) = e_pert(i) + div;
        else
            w_pert(i-3) = w_pert(i-3) + div;
        end
        
        [~, ~, we] = utils.CalculateWindInertial(v_tip, e_pert, 0, w_pert);
        
        sVw(i)  = (sqrt(we(1)^2 + we(2)^2) - Vw_nom) / div;
        sChi(i) = (atan2(we(2), we(1)) - chi_nom) / div;
        sWz(i)  = (we(3) - wz_nom) / div;
    end
end