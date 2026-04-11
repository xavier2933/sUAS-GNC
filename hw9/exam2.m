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
d_end = 103198;
d_start = 32280;

wx = we_all(:,1);
wy = we_all(:,2);

idx = d_start:1000:d_end-1;

lat_ds = data.lat(idx);
lon_ds = data.lon(idx);

wx_ds = wx(idx);
wy_ds = wy(idx);


n = length(data.lat);
t = data.Time;

%%%%% Q2.1
wx = we_all(d_start:d_end,1);
wy = we_all(d_start:d_end,2);
Vw = sqrt(wx.^2 + wy.^2);
chi_w = atan2(wy, wx);
data_lon = data.lon(d_start:d_end);
data_lat = data.lat(d_start:d_end);
idx = 1:700:length(data_lon);
t_chop = t(d_start:d_end);


figure()
subplot(3,1,1)
plot(t_chop, Vw)
ylabel("V_w")
subplot(3,1,2)
plot(t_chop, chi_w)
ylabel("\chi_w")
subplot(3,1,3)
plot(t_chop, we_all(d_start:d_end, 3))
ylabel("w_z")
xlabel("Time [minutes]")









%%%%%%%% Q2.2
euler_0 = [0;deg2rad(4);0];
w_0 = [18;deg2rad(-2);deg2rad(6)];
v_e_e_0 = [18;0;2];


div = 0.0001;
[vVw, vVchi, vWz] = get_sensitivities(v_e_e_0, euler_0, w_0, div)


% %% --- Gradient trajectory ---

% surface([data_lon data_lon], ...
%         [data_lat data_lat], ...
%         [t_chop t_chop], ...
%         [t_chop t_chop], ...
%         'EdgeColor', 'interp', ...
%         'FaceColor', 'none', ...
%         'LineWidth', 2);
% 
% colormap(turbo);
% cb = colorbar;
% cb.Label.String = 'Time';
% 
% xlabel('Longitude');
% ylabel('Latitude');
% title('Trajectory with Wind Vectors');
% 
% 
% wx = we_all(d_start:d_end,1);
% wy = we_all(d_start:d_end,2);
% 
% scale = 0.0003;
% 
% quiver(data_lon(idx), data_lat(idx), wx(idx)*scale, wy(idx)*scale, 0, 'm','LineWidth', 1);
% 
% legend('Trajectory', 'Wind');
% 
% 
% 
% %%%%%%%% stuff for exam 2















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