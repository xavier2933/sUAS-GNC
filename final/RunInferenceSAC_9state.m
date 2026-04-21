%% RunInferenceSAC_9state.m
% Runs the trained SAC agent and compares it against the SLC autopilot
% starting from the exact same initial conditions.
% Updated to support the 9-state observation space (includes prev_actions).

clear; close all;
addpath('..');
addpath('C:\Users\xavie\MATLAB\Projects\5128\hw7');

%% ─────────────────────────────────────────────
%%  CONFIG  ← change this to load a specific agent
%% ─────────────────────────────────────────────
% Leave '' to auto-detect the highest-numbered SAC checkpoint:
% Best to leave clear so it grabs the latest 9-state agent automatically
AGENT_FILE = 'saved_agents/agent_manual3.mat';

%% ─────────────────────────────────────────────
%%  1. Load Agent
%% ─────────────────────────────────────────────
if isempty(AGENT_FILE)
    agentFile = findBestSACAgent();
else
    agentFile = AGENT_FILE;
end
fprintf('Loading SAC agent from: %s\n', agentFile);
data = load(agentFile);
fnames = fieldnames(data);
if isfield(data, 'saved_agent')
    agent = data.saved_agent;
elseif isfield(data, 'agent')
    agent = data.agent;
else
    agent = data.(fnames{1});
end

%% ─────────────────────────────────────────────
%%  2. Create Environment & Fix Initial State
%% ─────────────────────────────────────────────
env = StraightLineEnv();

rng('shuffle');               % new random IC each run; both SAC & SLC share it
obs0   = reset(env);
state0 = env.aircraft_state; % shared IC for SAC and SLC

fprintf('Shared IC  —  cross_track: %.1f m | course_err: %.3f rad | h_err: %.1f m\n', ...
    obs0(1), obs0(2), obs0(3));

%% ─────────────────────────────────────────────
%%  3. SAC Agent Rollout
%% ─────────────────────────────────────────────
Ts       = env.Ts;
MaxSteps = env.MaxSteps;

rl_state  = zeros(12, MaxSteps+1);  rl_state(:,1)  = env.aircraft_state;
rl_obs    = zeros(9,  MaxSteps+1);  rl_obs(:,1)    = obs0;    % <--- Updated for 9 states
rl_action = zeros(3,  MaxSteps);
rl_reward = zeros(1,  MaxSteps);

obs    = obs0;
isDone = false;
k      = 0;

while ~isDone && k < MaxSteps
    % getAction samples from the stochastic Gaussian policy.
    action = cell2mat(getAction(agent, {obs}));

    [obs, reward, isDone, ~] = step(env, action);
    k = k + 1;

    rl_state(:,k+1)  = env.aircraft_state;
    rl_obs(:,k+1)    = obs;
    rl_action(:,k)   = action;
    rl_reward(k)     = reward;
end

N_rl      = k;
rl_time   = (0:N_rl) * Ts;
rl_state  = rl_state(:,  1:N_rl+1);
rl_obs    = rl_obs(:,    1:N_rl+1);
rl_action = rl_action(:, 1:N_rl);
rl_reward = rl_reward(   1:N_rl);

fprintf('SAC rollout complete — %d steps (%.0f s)  |  total reward: %.1f\n', ...
    N_rl, N_rl*Ts, sum(rl_reward));

%% ─────────────────────────────────────────────
%%  4. SLC Autopilot Rollout  (same IC, full EOM)
%% ─────────────────────────────────────────────
aircraft_parameters = utils.ttwistor();
V_trim        = env.V_trim;
pos_line      = env.pos_line;
dir_line      = env.dir_line;
kpath         = env.kpath;
chi_inf       = env.chi_inf;
wind_inertial = env.wind_inertial;

load('ttwistor_gains_slc', '-mat');
control_gain_struct.Ts                = Ts;
control_gain_struct.takeoff_height   = -999;
control_gain_struct.height_hold_limit = 99999;

slc_state = zeros(12, MaxSteps+1);
slc_state(:,1) = state0;
N_slc = MaxSteps;

for i = 1:MaxSteps
    TSPAN = Ts * [i-1, i];

    wind_body   = utils.TransformFromInertialToBody(wind_inertial, slc_state(4:6,i));
    air_rel     = slc_state(7:9,i) - wind_body;
    wind_angles = utils.AirRelativeVelocityVectorToWindAngles(air_rel);

    control_objectives = utils.StraightLineGuidance( ...
        pos_line, dir_line, slc_state(1:3,i), kpath, chi_inf, V_trim);

    [control_out, ~] = frewhw6utils.SimpleSLCAutopilot( ...
        Ts*(i-1), slc_state(:,i), wind_angles, control_objectives, control_gain_struct);

    [~, YOUT] = ode45(@(t,y) utils.AircraftEOM( ...
        t, y, control_out, wind_inertial, aircraft_parameters), ...
        TSPAN, slc_state(:,i), []);

    slc_state(:,i+1) = YOUT(end,:)';

    obs_i = computeObs(slc_state(:,i+1), pos_line, dir_line, V_trim, dir_line);
    if abs(obs_i(1)) > 300 || -slc_state(3,i+1) < 50
        N_slc = i;
        fprintf('SLC diverged at step %d\n', i);
        break;
    end
end

slc_state = slc_state(:, 1:N_slc+1);
slc_time  = (0:N_slc) * Ts;

slc_obs = zeros(4, N_slc+1);
for i = 1:N_slc+1
    slc_obs(:,i) = computeObs(slc_state(:,i), pos_line, dir_line, V_trim, dir_line);
end

fprintf('SLC rollout complete — %d steps (%.0f s)\n', N_slc, N_slc*Ts);
fprintf('Cross-track RMS  —  SLC: %.2f m  |  SAC: %.2f m\n', ...
    rms(slc_obs(1,:)), rms(rl_obs(1,:)));

%% ─────────────────────────────────────────────
%%  5. Recover actual SAC commands for plotting
%% ─────────────────────────────────────────────
chi_des    = atan2(dir_line(2), dir_line(1));
rl_chi_cmd = chi_des      + rl_action(1,:);   % [rad]
rl_h_cmd   = -pos_line(3) + rl_action(2,:);   % [m]
rl_Va_cmd  = V_trim       + rl_action(3,:);   % [m/s]
t_act      = rl_time(1:N_rl);

%% ─────────────────────────────────────────────
%%  PLOTS
%% ─────────────────────────────────────────────
des_line = [pos_line - 2000*dir_line, pos_line + 8000*dir_line];
C_slc = [0.18 0.45 0.71];   % blue  (SLC)
C_rl  = [0.13 0.63 0.30];   % green (SAC)

%% Figure 1 — 3D Trajectory
figure(1); clf;
plot3(des_line(1,:), des_line(2,:), -des_line(3,:), 'k--', 'LineWidth', 1.5); hold on;
plot3(slc_state(1,:), slc_state(2,:), -slc_state(3,:), 'Color', C_slc, 'LineWidth', 2);
plot3(rl_state(1,:),  rl_state(2,:),  -rl_state(3,:),  'Color', C_rl,  'LineWidth', 2);
plot3(state0(1), state0(2), -state0(3), 'ko', 'MarkerSize', 9, 'MarkerFaceColor', 'k');
xlabel('North (m)'); ylabel('East (m)'); zlabel('Height (m)');
title('3D Trajectory — SLC vs SAC');
legend('Desired line', 'SLC Autopilot', 'SAC Agent', 'Shared IC', 'Location', 'best');
grid on; view(3);

%% Figure 2 — Guidance Error States
figure(2); clf;

subplot(4,1,1);
plot(slc_time, slc_obs(1,:), 'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_obs(1,:),  'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
yline( 50, 'k:', 'LineWidth', 0.8); yline(-50, 'k:', 'LineWidth', 0.8);
ylabel('Cross-track (m)'); title('Guidance Error Comparison');
legend('SLC', 'SAC', 'Location', 'best'); grid on;

subplot(4,1,2);
plot(slc_time, slc_obs(2,:)*180/pi, 'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_obs(2,:)*180/pi,  'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
ylabel('Course error (°)'); grid on;

subplot(4,1,3);
plot(slc_time, slc_obs(3,:), 'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_obs(3,:),  'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
ylabel('Height error (m)'); grid on;

subplot(4,1,4);
plot(slc_time, slc_obs(4,:), 'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_obs(4,:),  'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
ylabel('V_a error (m/s)'); xlabel('Time (s)'); grid on;

%% Figure 3 — SAC Action Commands
figure(3); clf;

subplot(3,1,1);
plot(t_act, rl_chi_cmd*180/pi, 'Color', C_rl, 'LineWidth', 1.5); hold on;
yline(chi_des*180/pi, 'k--', sprintf('\\chi_{des}=%.1f°', chi_des*180/pi));
ylabel('\chi_{cmd} (°)'); title('SAC Agent Commands (actual values sent to autopilot)');
grid on;

subplot(3,1,2);
plot(t_act, rl_h_cmd, 'Color', C_rl, 'LineWidth', 1.5); hold on;
yline(-pos_line(3), 'k--', sprintf('h_{des}=%.0f m', -pos_line(3)));
ylabel('h_{cmd} (m)'); grid on;

subplot(3,1,3);
plot(t_act, rl_Va_cmd, 'Color', C_rl, 'LineWidth', 1.5); hold on;
yline(V_trim, 'k--', sprintf('V_{trim}=%.0f m/s', V_trim));
ylabel('V_{a,cmd} (m/s)'); xlabel('Time (s)'); grid on;

%% Figure 4 — SAC Reward Trace
figure(4); clf;
t_rwd = rl_time(1:N_rl);
yyaxis left;
plot(t_rwd, rl_reward, 'Color', [C_rl, 0.4], 'LineWidth', 1);
ylabel('Per-step reward');
yyaxis right;
plot(t_rwd, cumsum(rl_reward), 'Color', C_rl, 'LineWidth', 2);
ylabel('Cumulative reward');
xlabel('Time (s)');
title(sprintf('SAC Reward Trace  (total = %.1f over %d steps)', sum(rl_reward), N_rl));
grid on;

%% ─────────────────────────────────────────────
%%  Helper Functions
%% ─────────────────────────────────────────────

function agentFile = findBestSACAgent()
    % Search all sac_* subdirectories for the highest-numbered checkpoint.
    files = dir('saved_agents/sac_*/Agent*.mat');
    if ~isempty(files)
        nums = zeros(numel(files), 1);
        for k = 1:numel(files)
            tok = regexp(files(k).name, 'Agent(\d+)\.mat', 'tokens');
            if ~isempty(tok), nums(k) = str2double(tok{1}{1}); end
        end
        [~, idx] = max(nums);
        agentFile = fullfile(files(idx).folder, files(idx).name);
        return;
    end
    if exist('sac_slg_agent.mat', 'file')
        agentFile = 'sac_slg_agent.mat';
        return;
    end
    error('No SAC saved agents found. Run TrainSAC.m first.');
end

function obs = computeObs(state, pos_line, dir_line, V_trim, ~)
    p        = state(1:3);
    h_actual = -p(3);
    dp          = p - pos_line;
    dn          = norm(dp);
    cross_track = norm(dp - dot(dp, dir_line)*dir_line);
    if dn > 1e-6
        side = cross(dir_line, dp/dn);
        cross_track = sign(side(3)) * cross_track;
    end
    u   = state(7); v = state(8); psi = state(6);
    Vn  = u*cos(psi) - v*sin(psi);
    Ve  = u*sin(psi) + v*cos(psi);
    chi_actual = atan2(Ve, Vn);
    chi_des    = atan2(dir_line(2), dir_line(1));
    course_err = wrapToPi(chi_actual - chi_des);
    h_err  = h_actual - (-pos_line(3));
    Va_err = norm(state(7:9)) - V_trim;
    obs = [cross_track; course_err; h_err; Va_err];
end
