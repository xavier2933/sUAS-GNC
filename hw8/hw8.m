% Eric W. Frew
% ASEN 5128
% RunHW7.m
% Created: 3/2/23
%  
% This is a helper that students can use to complete HW 7. 
%
%
close all;% <========= Comment out this line and you can run this file multiple times and plot results together
clear all; 

addpath('..');

%%% Define parameters for specifying control law
SLC = 2;
FEED = 1;


%%% Set flags
ANIMATE_FLAG = 0; % <========= Set to 1 to show animation after simulation
CONTROL_FLAG = SLC; % <========= Set to control law to use (SLC or FEED)


%%% Aircraft parameter file
aircraft_parameters = utils.ttwistor();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Determine trim state and control inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

V_trim = 18;
h_trim = 1805;
gamma_trim = 0;
trim_definition = [V_trim; gamma_trim; h_trim];


%%% STUDENTS REPLACE THESE TWO FUNCTIONS WITH YOUR VERSIONS FROM HW 3/4
[aircraft_state_trim, control_input_trim, trim_variables, fval] = utils.CalculateTrim(trim_definition, aircraft_parameters);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Determine control gains to use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (CONTROL_FLAG==FEED)
    gains_file = 'ttwistor_gains_feed';
    fprintf(1, '\n==================================== \nAUTOPILOT: SLC with Feedforward\n \n')
else
    gains_file = 'ttwistor_gains_slc';
    fprintf(1, '\n ==================================== \nAUTOPILOT: Simple SLC\n \n')
end

load(gains_file)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Set input commands for autopilot.
%%%
%%% Note, STUDENTS may need to change these while tuning the autopilot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pos_line = [300;0;-h_trim];
% dir_line = [sqrt(2)/2;8*sqrt(2)/2;0];
dir_line = [sqrt(2)/2;8*sqrt(2)/2;0];

dir_line = dir_line / norm(dir_line);
kpath = 0.01;
chi_inf = 10 * pi/180;

des_line = [pos_line-1000*dir_line, pos_line + 8000*dir_line];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Set aircraft and simulation initial conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aircraft_state0 = aircraft_state_trim;

aircraft_state0(3,1) = -1705; %<------- CLIMB mode starts when aircraft reaches h = 1675
aircraft_state0(4,1) = 0*pi/180;

control_input0 = control_input_trim;

wind_inertial = [0;0;0];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Set simulation and control parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ts = .1;
Tfinal = 300;
control_gain_struct.Ts=Ts;

%%% iterate at control sample time
n_ind = Tfinal/Ts;

aircraft_array(:,1) = aircraft_state0;
control_array(:,1) = control_input0;
time_iter(1) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate and Plot first order
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[TFO, YFO] = ode45(@(t,y) utils.FirstOrderStraightLineGuidance(t, y,pos_line, dir_line, kpath, chi_inf, V_trim), [0,Tfinal], aircraft_state0(1:3,1),[]);
arr = 3

figure(8); hold on;
plot3(des_line(1,:), des_line(2,:), -des_line(3,:), 'k--'); hold on;

plot3(YFO(:,1), YFO(:,2), -YFO(:,3), 'g--');
xlabel('X');
ylabel('Y');
zlabel('Height');

for i = 1:length(TFO)
    chi_first(i) = atan2(YFO(i,2), YFO(i,1));
    Va_first(i) = V_trim;
end


figure(20);
subplot(311)
plot(TFO, YFO(:,1), 'g--'); hold on;
title('Position vs Time')
ylabel('X pos')

subplot(312)
plot(TFO, YFO(:,2), 'g--'); hold on;
ylabel('Y pos')

subplot(313)
plot(TFO, YFO(:,3), 'g--'); hold on;
ylabel('Z pos')
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate and Plot kinematic guidance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
state_guide = [0;0;0;0;-aircraft_state0(3,1);0;V_trim];
control_ob = [h_trim;0;deg2rad(30);0;V_trim]
control_objectives = control_ob;

aircraft_parameters.bc = 0.4;
aircraft_parameters.bcd = 1.5;
aircraft_parameters.bh = 0.08;
aircraft_parameters.bhd = 1.3;
aircraft_parameters.bva = 0.2;

[TGuide, YGuide] = ode45(@(t,y) utils.KinematicGuidanceModel(t, y, [0;0;0],control_ob, aircraft_parameters),[0 Tfinal],state_guide, []);
figure;
subplot(311)
plot(TGuide, YGuide(:,3), 'r-.'); hold on;
subplot(312)
plot(TGuide, YGuide(:,5), 'r-.'); hold on;
subplot(313)
plot(TGuide, YGuide(:,7), 'r-.'); hold on;



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate straightline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
guide_array(:,1) = state_guide;
control_guide_array(:,1) = [-aircraft_state0(3,1);0;0;0;V_trim];
time_iter_guide(1) = 0;
for i=1:n_ind

    TSPAN = Ts*[i-1 i];

    wind_array_guide(:,i) = wind_inertial;
    % 
    % wind_body = utils.TransformFromInertialToBody(wind_inertial, aircraft_array(4:6,i));
    % air_rel_vel_body = aircraft_array(7:9,i) - wind_body;
    % wind_angles(:,i) = utils.AirRelativeVelocityVectorToWindAngles(air_rel_vel_body);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Guidance level commands
    %%%
    % control_objectives(1) = 1805;
    % control_objectives(2) = 0;
    % control_objectives(3) = 0;
    % control_objectives(4) = 18/600;
    % control_objectives(5) = 18;
    %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % STUDENTS WRITE THIS FUNCTION
    control_objectives = utils.StraightLineGuidance(pos_line, dir_line, [guide_array(1:2,i); guide_array(5,i)], kpath, chi_inf, V_trim);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Aircraft dynamics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [TOUT2, YOUT2] = ode45(@(t,y) utils.KinematicGuidanceModel(t, y, [0;0;0], control_objectives, aircraft_parameters), TSPAN, guide_array(:,i), []);

    guide_array(:,i+1) = YOUT2(end,:)';
    time_iter_guide(i+1) = TOUT2(end);
    wind_array_guide(:,i+1) = wind_inertial;
    control_guide_array(:,i+1) = control_objectives;
end

%% --- Comparison Plots ---

% --- Figure 8: 3D Path ---
figure(8); hold on;
% guide_array indices: 1=Pn, 2=Pe, 5=h
plot3(guide_array(1,:), guide_array(2,:), guide_array(5,:), 'r', 'LineWidth', 1.5);
legend('Desired Line', 'First-Order', 'Kinematic Guidance');

% --- Figure 20: Position vs Time ---
figure(20);
subplot(311); hold on;
plot(time_iter_guide, guide_array(1,:), 'r');
ylabel('X pos (N)')

subplot(312); hold on;
plot(time_iter_guide, guide_array(2,:), 'r');
ylabel('Y pos (E)')

subplot(313); hold on;
% Note: YFO uses Pd (Down), so we plot -YFO(:,3) for height comparison
% Or if YFO was already Height, just use guide_array(5,:)
plot(time_iter_guide, guide_array(5,:), 'r');
ylabel('Height (Z)')
legend('First-Order', 'Kinematic');

% --- New Figure: Guidance States (Chi and Va) ---
figure(21); clf;
subplot(211); hold on;
plot(TFO, chi_first*180/pi, 'g--'); % First order
plot(time_iter_guide, guide_array(3,:)*180/pi, 'r'); % Kinematic
ylabel('\chi [deg]'); title('Course Comparison');

subplot(212); hold on;
plot(TFO, Va_first, 'g--');
plot(time_iter_guide, guide_array(7,:), 'r');
ylabel('V_a [m/s]'); xlabel('Time [s]');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate full system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:n_ind

    TSPAN = Ts*[i-1 i];

    wind_array(:,i) = wind_inertial;

    wind_body = utils.TransformFromInertialToBody(wind_inertial, aircraft_array(4:6,i));
    air_rel_vel_body = aircraft_array(7:9,i) - wind_body;
    wind_angles(:,i) = utils.AirRelativeVelocityVectorToWindAngles(air_rel_vel_body);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Guidance level commands
    %%%
    % control_objectives(1) = 1805;
    % control_objectives(2) = 0;
    % control_objectives(3) = 0;
    % control_objectives(4) = 18/600;
    % control_objectives(5) = 18;
    %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % STUDENTS WRITE THIS FUNCTION
    control_objectives = utils.StraightLineGuidance(pos_line, dir_line, aircraft_array(1:3,i), kpath, chi_inf, V_trim); 

    %control_gain_struct.Kp_course_rate=0.0; %<============== Uncomment if your guidance algorithm does not give a command course angle, i.e. only gives commanded course rate
    %control_gain_struct.Kff_course_rate = 1.0; %<============== Uncomment if your guidance algorithm does not give a command course rate

    %control_objectives = [100; 0; 45*pi/180; 0; V_trim]; %<============== Comment out when OrbitGuidance is complete

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Autopilot
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (CONTROL_FLAG==FEED)
        [control_out, x_c_out] = hw7utils.SLCWithFeedForwardAutopilot(Ts*(i-1), aircraft_array(:,i), wind_angles(:,i), control_objectives, control_gain_struct);
    else
        [control_out, x_c_out] = frewhw6utils.SimpleSLCAutopilot(Ts*(i-1), aircraft_array(:,i), wind_angles(:,i), control_objectives, control_gain_struct);
    end

    control_array(:,i) = control_out;
    x_command(:,i) = x_c_out;
    x_command(5,i) = trim_variables(1);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Aircraft dynamics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [TOUT2,YOUT2] = ode45(@(t,y) utils.AircraftEOM(t,y,control_array(:,i),wind_inertial,aircraft_parameters),TSPAN,aircraft_array(:,i),[]);


    aircraft_array(:,i+1) = YOUT2(end,:)';
    time_iter(i+1) = TOUT2(end);
    wind_array(:,i+1) = wind_inertial;
    control_array(:,i+1) = control_array(:,i);
    x_command(:,i+1) = x_command(:,i);
end

%% --- Comparison Plots (Fidelity: First-Order vs Kinematic vs Full EOM) ---

% --- Figure 8: 3D Path ---
figure(8); hold on;
% Full System Plot (Indices: 1=Pn, 2=Pe, 3=Pd)
% Note: We use -aircraft_array(3,:) because Pd is Down and we want Height
plot3(aircraft_array(1,:), aircraft_array(2,:), -aircraft_array(3,:), 'b', 'LineWidth', 1.5);
legend('Desired Line', 'First-Order', 'Kinematic Guidance', 'Full EOM');
grid on; view(3);

% --- Figure 20: Position vs Time ---
figure(20);
subplot(311); hold on;
plot(time_iter, aircraft_array(1,:), 'b');
ylabel('X pos (N)')

subplot(312); hold on;
plot(time_iter, aircraft_array(2,:), 'b');
ylabel('Y pos (E)')

subplot(313); hold on;
% Plotting Height (Negative of Downward position)
plot(time_iter, -aircraft_array(3,:), 'b');
ylabel('Height (Z)')
legend('First-Order', 'Kinematic', 'Full EOM');

% --- Figure 21: Guidance & State States (Chi, Height, and Va) ---
figure(21); clf;

% --- SUBPLOT 1: Course Angle ---
subplot(311); hold on;
chi_first_fixed = atan2(diff(YFO(:,2)), diff(YFO(:,1))); % Gemini fix. finds velocity for proper course angle calc
chi_first_fixed = [chi_first_fixed(1); chi_first_fixed]; % Match array length
plot(TFO, chi_first_fixed*180/pi, 'g--'); 

% 2. Kinematic
plot(time_iter_guide, guide_array(3,:)*180/pi, 'r'); 

psi_blue = aircraft_array(6,:);
u_blue   = aircraft_array(7,:);
v_blue   = aircraft_array(8,:);
V_n_blue = u_blue .* cos(psi_blue) - v_blue .* sin(psi_blue);
V_e_blue = u_blue .* sin(psi_blue) + v_blue .* cos(psi_blue);
chi_full_fixed = atan2(V_e_blue, V_n_blue);
plot(time_iter, chi_full_fixed*180/pi, 'b'); 

ylabel('\chi [deg]'); title('Guidance State Comparison');
grid on;

% --- SUBPLOT 2: Height ---
subplot(312); hold on;
plot(TFO, -YFO(:,3), 'g--');
plot(time_iter_guide, guide_array(5,:), 'r');
plot(time_iter, -aircraft_array(3,:), 'b');
ylabel('Height [m]');
grid on;

% --- SUBPLOT 3: Airspeed ---
subplot(313); hold on;
plot(TFO, ones(size(TFO))*V_trim, 'g--'); 
plot(time_iter_guide, guide_array(7,:), 'r'); 
plot(time_iter, aircraft_array(7,:), 'b');
ylabel('V_a [m/s]'); xlabel('Time [s]');
legend('First-Order', 'Kinematic', 'Full EOM', 'Location', 'best');
grid on;