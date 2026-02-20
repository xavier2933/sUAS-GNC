function [flight_angles] = FlightPathAnglesFromState(aircraft_state)

    phi = aircraft_state(4);
    theta = aircraft_state(5);
    psi = aircraft_state(6);

    % 2. Extract Euler Angles (indices 7-9)
    u   = aircraft_state(7);
    v = aircraft_state(8);
    w   = aircraft_state(9);


    res = utils.RotationMatrix321([phi;theta;psi]) * [u;v;w];

    u_e = res(1);
    v_e = res(2);
    w_e = res(3);

    V_g = sqrt(u_e^2 + v_e^2 + w_e^2);
    chi = -atan2(v_e, u_e);
    gamma = atan2(w_e, sqrt(u_e^2 + v_e^2));

    flight_angles = [V_g;chi;gamma];

end