clear; clc; close all
addpath('..');

data = load('RaavenWindData.mat')

n = size(data.Va, 1);

% Preallocate
Va_all   = zeros(n, 3);
Va_e_all = zeros(n, 3);
we_all   = zeros(n, 3);

for i = 1:n
    v_tip = data.aircraft_velocity_inertial(:,i);

    euler_angles = [ ...
        deg2rad(data.roll(i));
        deg2rad(data.pitch(i));
        deg2rad(data.yaw(i)) ];

    angular_rates = 0;

    wind_angles  = [ ...
        data.Va(i);
        deg2rad(data.beta(i));
        deg2rad(data.alpha(i)) ];

    [Va, Va_e, we] = utils.CalculateWindInertial( ...
        v_tip, euler_angles, angular_rates, wind_angles);
    Va_all(i, :)   = Va(:)';
    Va_e_all(i, :) = Va_e(:)';
    we_all(i, :)   = we(:)';
end

t = 1:n;

wx = we_all(:,1);
wy = we_all(:,2);

idx = 1:1000:length(data.lat);

lat_ds = data.lat(idx);
lon_ds = data.lon(idx);

wx_ds = wx(idx);
wy_ds = wy(idx);

figure; hold on; grid on;

n = length(data.lat);
t = data.Time;

%% --- Gradient trajectory ---
surface([data.lon data.lon], ...
        [data.lat data.lat], ...
        [t t], ...
        [t t], ...
        'EdgeColor', 'interp', ...
        'FaceColor', 'none', ...
        'LineWidth', 2);

colormap(turbo);
cb = colorbar;
cb.Label.String = 'Time';

xlabel('Longitude');
ylabel('Latitude');
title('Trajectory with Wind Vectors');

idx = 1:700:n;

wx = we_all(:,1);
wy = we_all(:,2);

scale = 0.0003;

quiver(data.lon(idx), data.lat(idx), wx(idx)*scale, wy(idx)*scale, 0, 'm','LineWidth', 1);

legend('Trajectory', 'Wind');



% %% --- Figure 1: Airspeed (Body Frame) ---
% figure('Name', 'Airspeed - Body Frame', 'NumberTitle', 'off');
% labels_b = {'u (forward)', 'v (lateral)', 'w (vertical)'};
% for i = 1:3
%     subplot(3, 1, i);
%     plot(t, Va_all(:, i), 'Color', [0 0.447 0.741]); % Blue
%     ylabel(['V_{a,b} ', labels_b{i}, ' (m/s)']);
%     grid on;
%     if i == 1, title('Airspeed Components: Body Frame'); end
% end
% xlabel('Time / Samples');
% 
% %% --- Figure 2: Airspeed (Inertial Frame) ---
% figure('Name', 'Airspeed - Inertial Frame', 'NumberTitle', 'off');
% labels_e = {'X (North)', 'Y (East)', 'Z (Down)'};
% for i = 1:3
%     subplot(3, 1, i);
%     plot(t, Va_e_all(:, i), 'Color', [0.85 0.325 0.098]); % Orange
%     ylabel(['V_{a,e} ', labels_e{i}, ' (m/s)']);
%     grid on;
%     if i == 1, title('Airspeed Components: Inertial Frame'); end
% end
% xlabel('Time / Samples');
% 
% %% --- Figure 3: Wind (Inertial Frame) ---
% figure('Name', 'Wind - Inertial Frame', 'NumberTitle', 'off');
% labels_w = {'W_n', 'W_e', 'W_d'};
% for i = 1:3
%     subplot(3, 1, i);
%     plot(t, we_all(:, i), 'Color', [0.466 0.674 0.188]); % Green
%     ylabel(['Wind ', labels_w{i}, ' (m/s)']);
%     grid on;
%     if i == 1, title('Wind Components: Inertial Frame'); end
% end
% xlabel('Time / Samples');
% 
% % Link the x-axes of all subplots across all figures for synchronized zooming
% all_axes = findall(0, 'type', 'axes');
% linkaxes(all_axes, 'x');