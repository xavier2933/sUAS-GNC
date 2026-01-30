function PlotSimulation(time, aircraft_state_array, control_input_array, col)

    % -------- Position (x y z) --------
    figure(1)

    subplot(311); hold on;
    plot(time, aircraft_state_array(1,:), col);
    ylabel('x (m)');
    title('Inertial Position');

    subplot(312); hold on;
    plot(time, aircraft_state_array(2,:), col);
    ylabel('y (m)');

    subplot(313); hold on;
    plot(time, aircraft_state_array(3,:), col);
    ylabel('z (m)');
    xlabel('Time (s)');

    % -------- Euler angles (phi theta psi) --------
    figure(2)

    subplot(311); hold on;
    plot(time, aircraft_state_array(4,:), col);
    ylabel('\phi (rad)');
    title('Euler Angles');

    subplot(312); hold on;
    plot(time, aircraft_state_array(5,:), col);
    ylabel('\theta (rad)');

    subplot(313); hold on;
    plot(time, aircraft_state_array(6,:), col);
    ylabel('\psi (rad)');
    xlabel('Time (s)');

    % -------- Body velocities (u v w) --------
    figure(3)

    subplot(311); hold on;
    plot(time, aircraft_state_array(7,:), col);
    ylabel('u (m/s)');
    title('Body-Frame Velocities');

    subplot(312); hold on;
    plot(time, aircraft_state_array(8,:), col);
    ylabel('v (m/s)');

    subplot(313); hold on;
    plot(time, aircraft_state_array(9,:), col);
    ylabel('w (m/s)');
    xlabel('Time (s)');

    % -------- Body rates (p q r) --------
    figure(4)

    subplot(311); hold on;
    plot(time, aircraft_state_array(10,:), col);
    ylabel('p (rad/s)');
    title('Body Rates');

    subplot(312); hold on;
    plot(time, aircraft_state_array(11,:), col);
    ylabel('q (rad/s)');

    subplot(313); hold on;
    plot(time, aircraft_state_array(12,:), col);
    ylabel('r (rad/s)');
    xlabel('Time (s)');

    % -------- Control inputs --------
    figure(5)

    subplot(411); hold on;
    plot(time, control_input_array(1,:), col);
    ylabel('\delta_e');

    subplot(412); hold on;
    plot(time, control_input_array(2,:), col);
    ylabel('\delta_a');

    subplot(413); hold on;
    plot(time, control_input_array(3,:), col);
    ylabel('\delta_r');

    subplot(414); hold on;
    plot(time, control_input_array(4,:), col);
    ylabel('\delta_t');
    xlabel('Time (s)');

    title('Control Inputs');

    % -------- 3D trajectory --------
    figure(6); hold on;
    plot3(aircraft_state_array(1,:), ...
          aircraft_state_array(2,:), ...
          aircraft_state_array(3,:), col);
    
    % Add start (green) and end (red) markers
    plot3(aircraft_state_array(1,1), ...
          aircraft_state_array(2,1), ...
          aircraft_state_array(3,1), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    
    plot3(aircraft_state_array(1,end), ...
          aircraft_state_array(2,end), ...
          aircraft_state_array(3,end), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');

    xlabel('X inertial (m)');
    ylabel('Y inertial (m)');
    zlabel('Z inertial (m)');
    title('Aircraft 3D Trajectory');
    grid on;
    axis equal;
    view(3)

    % ------- MEGA PLOT -------
    figure(7); clf
    tiledlayout(5,4,'TileSpacing','compact','Padding','compact');
    
    % -------- Position --------
    nexttile; plot(time, aircraft_state_array(1,:), col); ylabel('x (m)'); title('Position')
    nexttile; plot(time, aircraft_state_array(2,:), col); ylabel('y (m)')
    nexttile; plot(time, aircraft_state_array(3,:), col); ylabel('z (m)')
    nexttile; axis off
    
    % -------- Euler Angles --------
    nexttile; plot(time, aircraft_state_array(4,:), col); ylabel('\phi (rad)'); title('Euler Angles')
    nexttile; plot(time, aircraft_state_array(5,:), col); ylabel('\theta (rad)')
    nexttile; plot(time, aircraft_state_array(6,:), col); ylabel('\psi (rad)')
    nexttile; axis off
    
    % -------- Body Velocities --------
    nexttile; plot(time, aircraft_state_array(7,:), col); ylabel('u (m/s)'); title('Body Velocities')
    nexttile; plot(time, aircraft_state_array(8,:), col); ylabel('v (m/s)')
    nexttile; plot(time, aircraft_state_array(9,:), col); ylabel('w (m/s)')
    nexttile; axis off
    
    % -------- Body Rates --------
    nexttile; plot(time, aircraft_state_array(10,:), col); ylabel('p (rad/s)'); title('Body Rates')
    nexttile; plot(time, aircraft_state_array(11,:), col); ylabel('q (rad/s)')
    nexttile; plot(time, aircraft_state_array(12,:), col); ylabel('r (rad/s)')
    nexttile; axis off
    
    % -------- Controls --------
    nexttile; plot(time, control_input_array(1,:), col); ylabel('\delta_e'); title('Controls')
    nexttile; plot(time, control_input_array(2,:), col); ylabel('\delta_a')
    nexttile; plot(time, control_input_array(3,:), col); ylabel('\delta_r')
    nexttile; plot(time, control_input_array(4,:), col); ylabel('\delta_t'); xlabel('Time (s)')


end