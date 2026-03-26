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

t = 1:n; % or use actual time vector if available

wx = we_all(:,1);
wy = we_all(:,2);

idx = 1:1000:length(data.lat);

lat_ds = data.lat(idx);
lon_ds = data.lon(idx);

wx_ds = wx(idx);
wy_ds = wy(idx);

figure; hold on; grid on;

n = length(data.lat);
t = data.Time;   % or (1:n)

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

%% --- Wind vectors (downsampled) ---
idx = 1:1000:n;

% Extract horizontal wind
wx = we_all(:,1);
wy = we_all(:,2);

% Scale for visibility (tune this!)
scale = 0.0005;

quiver(data.lon(idx), data.lat(idx), ...
       wx(idx)*scale, wy(idx)*scale, ...
       0, ...                % disable auto scaling
       'm', ...              % black arrows for contrast
       'LineWidth', 1);

legend('Trajectory', 'Wind');



% figure;
% plot(t, Va_all);
% title('V_a (Body Frame)');
% legend('x','y','z');
% 
% figure;
% plot(t, Va_e_all);
% title('V_a (Inertial Frame)');
% legend('x','y','z');
% 
% figure;
% plot(t, we_all);
% title('Wind (Inertial Frame)');
% legend('x','y','z');