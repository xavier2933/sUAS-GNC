%% RunInferenceV2.m
% Runs the trained PPO agent and compares it against the SLC autopilot
% starting from the exact same initial conditions.
%
% ── Agent loading priority ────────────────────────────────────────────────
%   1. AGENT_FILE (set below) — explicit override
%   2. saved_agents/run_*/stage4/Agent<N>.mat — highest-numbered checkpoint
%      from the final (hardest) curriculum stage of the most recent run
%   3. Lower stages (stage3 → stage1) of the most recent run
%   4. ppo_slg_agent.mat — end-of-run flat save from TrainPPO.m
%
% ── Compatibility note ────────────────────────────────────────────────────
%   Requires an agent trained with the UPDATED StraightLineEnv:
%     • Observations normalised to [-1, 1] before being returned
%     • Exponential reward function   (no quadratic penalty)
%     • 128×128 actor/critic networks
%   Agents from earlier runs used raw observations and will behave incorrectly.

clear; close all;
addpath('..');
addpath('C:\Users\xavie\MATLAB\Projects\5128\hw7');

%% ─────────────────────────────────────────────
%%  CONFIG  ← edit here
%% ─────────────────────────────────────────────
% AGENT_FILE    = 'saved_agents/run_0423_1023/stage1/Agent497.mat';
% AGENT_FILE = 'saved_agents/run_0423_2200/stage2/Agent116.mat'; % GOOD
AGENT_FILE = 'saved_agents/run_0424_1942/stage1/Agent508.mat';

IC_DIFFICULTY = 0.2;  % IC spread for inference: 0=easy (~15 m), 1=hard (~135 m)

% --- AIRCRAFT MASS OVERRIDE ---
OVERRIDE_MASS = false; % Set to true to test a different mass
TEST_MASS_KG  = 4.5;   % [kg] New mass (nominal ttwistor is 5.74 kg)

% --- INITIAL CONDITION TWEAKS ---
USE_FIXED_SEED = true; % Set to true for repeatable randomly-generated ICs
FIXED_SEED_VAL = 69;   % The random seed to use 69 for below line, 42 for above

% --- WIND OVERRIDE ---
INERTIAL_WIND  = [0; 0; 0]; % [North, East, Down] m/s steady wind (e.g., [0; 5; 0] for crosswind)

%% ─────────────────────────────────────────────
%%  1. Load Agent
%% ─────────────────────────────────────────────
if isempty(AGENT_FILE)
    agentFile = findBestAgent();
else
    agentFile = AGENT_FILE;
end
fprintf('Loading agent from: %s\n', agentFile);
data   = load(agentFile);
fnames = fieldnames(data);
if isfield(data, 'saved_agent')
    agent = data.saved_agent;
elseif isfield(data, 'agent')
    agent = data.agent;
else
    agent = data.(fnames{1});   % take whatever field is present
end

%% ─────────────────────────────────────────────
%%  2. Create Environment & Fix Initial State
%% ─────────────────────────────────────────────
env = StraightLineEnv();
env.difficulty = IC_DIFFICULTY;   % controls IC spread (matches training stage)
env.wind_inertial = INERTIAL_WIND; % apply steady wind

if OVERRIDE_MASS
    env.aircraft_parameters.m = TEST_MASS_KG;
    env.aircraft_parameters.W = TEST_MASS_KG * env.aircraft_parameters.g;
    % Recalculate trim for the new mass
    trim_def = [env.V_trim; 0; 1805];
    [env.aircraft_state_trim, env.control_input_trim, ...
     env.trim_variables, ~] = utils.CalculateTrim(trim_def, env.aircraft_parameters);
    fprintf('OVERRIDING MASS: Set to %.2f kg in PPO Environment (recalculated trim)\n', TEST_MASS_KG);
end

if USE_FIXED_SEED
    rng(FIXED_SEED_VAL);          % deterministic random IC for comparisons
else
    rng('shuffle');               % fresh random IC each run
end
obs0   = reset(env);              % returns NORMALISED 6-element obs
state0 = env.aircraft_state;     % shared IC for RL and SLC

% === CUSTOM INITIAL STATE OVERRIDE ========================================
% To manually set the initial state, uncomment and edit the block below:
%{
state0(1)  = 0;          % pn    (North position, m)
state0(2)  = 100;        % pe    (East position, m)       <-- e.g. cross-track
state0(3)  = -1805;      % pd    (Down position, m)       <-- negative altitude
state0(4)  = 0;          % phi   (Roll angle, rad)
state0(5)  = 0;          % theta (Pitch angle, rad)
state0(6)  = pi/4;       % psi   (Yaw angle, rad)         <-- course error
state0(7)  = 18;         % u     (Body x velocity, m/s)   <-- airspeed
state0(8)  = 0;          % v     (Body y velocity, m/s)
state0(9)  = 0;          % w     (Body z velocity, m/s)
state0(10) = 0;          % p     (Roll rate, rad/s)
state0(11) = 0;          % q     (Pitch rate, rad/s)
state0(12) = 0;          % r     (Yaw rate, rad/s)

% Apply override to the environment
env.aircraft_state = state0;

% Recompute the normalised observation fed into the agent for step 0
raw_obs0 = computeObs(state0, env.pos_line, env.dir_line, env.V_trim);
obs0     = [raw_obs0(1)/500; raw_obs0(2)/pi; raw_obs0(3)/200; raw_obs0(4)/10; 0; 0];
%}
% ==========================================================================

% De-normalise for display (obs0 is in [-1,1])
fprintf('Shared IC  —  cross_track: %.1f m | course_err: %.3f rad | h_err: %.1f m\n', ...
    obs0(1)*500, obs0(2)*pi, obs0(3)*200);

%% ─────────────────────────────────────────────
%%  3. PPO Agent Rollout
%% ─────────────────────────────────────────────
Ts       = env.Ts;
MaxSteps = env.MaxSteps;

pos_line = env.pos_line;
dir_line = env.dir_line;
V_trim   = env.V_trim;

% Pre-allocate.  rl_raw_obs stores UN-normalised values (m, rad) for plots.
rl_state   = zeros(12, MaxSteps+1);   rl_state(:,1)   = env.aircraft_state;
rl_raw_obs = zeros(4,  MaxSteps+1);   % [cross_track; course_err; h_err; Va_err]
rl_action  = zeros(3,  MaxSteps);
rl_reward  = zeros(1,  MaxSteps);

rl_raw_obs(:,1) = computeObs(state0, pos_line, dir_line, V_trim);

obs    = obs0;    % normalised obs fed to the agent
isDone = false;
k      = 0;

while ~isDone && k < MaxSteps
    % getAction samples stochastically from the Gaussian policy.
    % For a deterministic (mean-only) rollout call getActionInfo instead.
    action = cell2mat(getAction(agent, {obs}));

    [obs, reward, isDone, ~] = step(env, action);
    k = k + 1;

    rl_state(:,k+1)   = env.aircraft_state;
    rl_raw_obs(:,k+1) = computeObs(env.aircraft_state, pos_line, dir_line, V_trim);
    rl_action(:,k)    = action;
    rl_reward(k)      = reward;
end

N_rl       = k;
rl_time    = (0:N_rl) * Ts;
rl_state   = rl_state(:,   1:N_rl+1);
rl_raw_obs = rl_raw_obs(:, 1:N_rl+1);
rl_action  = rl_action(:,  1:N_rl);
rl_reward  = rl_reward(     1:N_rl);

fprintf('RL rollout complete — %d steps (%.0f s)  |  total reward: %.1f\n', ...
    N_rl, N_rl*Ts, sum(rl_reward));

%% ─────────────────────────────────────────────
%%  4. SLC Autopilot Rollout  (same IC, full EOM)
%% ─────────────────────────────────────────────
aircraft_parameters = utils.ttwistor();

if OVERRIDE_MASS
    aircraft_parameters.m = TEST_MASS_KG;
    aircraft_parameters.W = TEST_MASS_KG * aircraft_parameters.g;
    fprintf('OVERRIDING MASS: Set to %.2f kg in SLC Autopilot\n', TEST_MASS_KG);
end

kpath         = env.kpath;
chi_inf       = env.chi_inf;
wind_inertial = env.wind_inertial;

load('ttwistor_gains_feed', '-mat');
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

    [control_out, ~] = hw7utils.SLCWithFeedForwardAutopilot( ...
        Ts*(i-1), slc_state(:,i), wind_angles, control_objectives, control_gain_struct);

    [~, YOUT] = ode45(@(t,y) utils.AircraftEOM( ...
        t, y, control_out, wind_inertial, aircraft_parameters), ...
        TSPAN, slc_state(:,i), []);

    slc_state(:,i+1) = YOUT(end,:)';

    obs_i = computeObs(slc_state(:,i+1), pos_line, dir_line, V_trim);
    if abs(obs_i(1)) > 200 || -slc_state(3,i+1) < 50
        N_slc = i;
        fprintf('SLC diverged at step %d\n', i);
        break;
    end
end

slc_state = slc_state(:, 1:N_slc+1);
slc_time  = (0:N_slc) * Ts;

slc_obs = zeros(4, N_slc+1);
for i = 1:N_slc+1
    slc_obs(:,i) = computeObs(slc_state(:,i), pos_line, dir_line, V_trim);
end

fprintf('SLC rollout complete — %d steps (%.0f s)\n', N_slc, N_slc*Ts);
fprintf('Cross-track RMS  —  SLC: %.2f m  |  PPO: %.2f m\n', ...
    rms(slc_obs(1,:)), rms(rl_raw_obs(1,:)));

%% ─────────────────────────────────────────────
%%  5. Recover actual PPO commands for plotting
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
C_slc = [0.18 0.45 0.71];   % blue
C_rl  = [0.84 0.15 0.15];   % red

%% Figure 1 — 3D Trajectory
figure(1); clf;
plot3(des_line(1,:), des_line(2,:), -des_line(3,:), 'k--', 'LineWidth', 1.5); hold on;
plot3(slc_state(1,:), slc_state(2,:), -slc_state(3,:), 'Color', C_slc, 'LineWidth', 2);
plot3(rl_state(1,:),  rl_state(2,:),  -rl_state(3,:),  'Color', C_rl,  'LineWidth', 2);
plot3(state0(1), state0(2), -state0(3), 'ko', 'MarkerSize', 9, 'MarkerFaceColor', 'k');
xlabel('North (m)'); ylabel('East (m)'); zlabel('Height (m)');
title('3D Trajectory — SLC vs PPO');
legend('Desired line', 'SLC Autopilot', 'PPO Agent', 'Shared IC', 'Location', 'best');
grid on; view(3);

%% Figure 2 — Guidance Error States
figure(2); clf;

subplot(4,1,1);
plot(slc_time, slc_obs(1,:),         'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_raw_obs(1,:),      'Color', C_rl,  'LineWidth', 1.5);
yline(0,   'k--', 'LineWidth', 0.8);
yline( 50, 'k:',  'LineWidth', 0.8); yline(-50, 'k:', 'LineWidth', 0.8);
ylabel('Cross-track (m)'); title('Guidance Error Comparison');
legend('SLC', 'PPO', 'Location', 'best'); grid on;

subplot(4,1,2);
plot(slc_time, slc_obs(2,:)*180/pi,    'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_raw_obs(2,:)*180/pi, 'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
ylabel('Course error (°)'); grid on;

subplot(4,1,3);
plot(slc_time, slc_obs(3,:),      'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_raw_obs(3,:),   'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
ylabel('Height error (m)'); grid on;

subplot(4,1,4);
plot(slc_time, slc_obs(4,:),      'Color', C_slc, 'LineWidth', 1.5); hold on;
plot(rl_time,  rl_raw_obs(4,:),   'Color', C_rl,  'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.8);
ylabel('V_a error (m/s)'); xlabel('Time (s)'); grid on;

%% Figure 3 — PPO Action Commands
figure(3); clf;

subplot(3,1,1);
plot(t_act, rl_chi_cmd*180/pi, 'Color', C_rl, 'LineWidth', 1.5); hold on;
yline(chi_des*180/pi, 'k--', sprintf('\\chi_{des}=%.1f°', chi_des*180/pi));
ylabel('\chi_{cmd} (°)'); title('PPO Agent Commands (actual values sent to autopilot)');
grid on;

subplot(3,1,2);
plot(t_act, rl_h_cmd, 'Color', C_rl, 'LineWidth', 1.5); hold on;
yline(-pos_line(3), 'k--', sprintf('h_{des}=%.0f m', -pos_line(3)));
ylabel('h_{cmd} (m)'); grid on;

subplot(3,1,3);
plot(t_act, rl_Va_cmd, 'Color', C_rl, 'LineWidth', 1.5); hold on;
yline(V_trim, 'k--', sprintf('V_{trim}=%.0f m/s', V_trim));
ylabel('V_{a,cmd} (m/s)'); xlabel('Time (s)'); grid on;

%% Figure 4 — PPO Reward Trace
figure(4); clf;
t_rwd = rl_time(1:N_rl);
yyaxis left;
plot(t_rwd, rl_reward, 'Color', [C_rl, 0.4], 'LineWidth', 1);
ylabel('Per-step reward');
yyaxis right;
plot(t_rwd, cumsum(rl_reward), 'Color', C_rl, 'LineWidth', 2);
ylabel('Cumulative reward');
xlabel('Time (s)');
title(sprintf('PPO Reward Trace  (total = %.1f over %d steps)', sum(rl_reward), N_rl));
grid on;

%% ─────────────────────────────────────────────
%%  Helper Functions
%% ─────────────────────────────────────────────

function agentFile = findBestAgent()
% Search the most recent run's stage dirs (4 → 1), then the run root,
% then a flat saved_agents/ folder, then ppo_slg_agent.mat.
    stageDirs = {'stage4', 'stage3', 'stage2', 'stage1'};

    runDirs = dir('saved_agents/run_*');
    runDirs = runDirs([runDirs.isdir]);
    if ~isempty(runDirs)
        [~, idx] = sort({runDirs.name}, 'descend');   % most recent first
        runDirs  = runDirs(idx);
    end

    for r = 1:numel(runDirs)
        % Check stage subdirs first (curriculum training layout)
        for sd = 1:numel(stageDirs)
            searchPath = fullfile('saved_agents', runDirs(r).name, stageDirs{sd});
            files = dir(fullfile(searchPath, 'Agent*.mat'));
            if ~isempty(files)
                agentFile = pickHighestNumbered(files, searchPath);
                return;
            end
        end
        % Fallback: files directly in the run dir (non-curriculum layout)
        files = dir(fullfile('saved_agents', runDirs(r).name, 'Agent*.mat'));
        if ~isempty(files)
            agentFile = pickHighestNumbered(files, fullfile('saved_agents', runDirs(r).name));
            return;
        end
    end

    % Flat saved_agents/ directory
    files = dir('saved_agents/Agent*.mat');
    if ~isempty(files)
        agentFile = pickHighestNumbered(files, 'saved_agents');
        return;
    end

    % Last resort: end-of-run flat save
    if exist('ppo_slg_agent.mat', 'file')
        agentFile = 'ppo_slg_agent.mat';
        return;
    end
    error('No saved agents found. Run TrainPPO.m first.');
end

function agentFile = pickHighestNumbered(files, basePath)
% Return the path of the Agent<N>.mat with the highest N.
    nums = zeros(numel(files), 1);
    for k = 1:numel(files)
        tok = regexp(files(k).name, 'Agent(\d+)\.mat', 'tokens');
        if ~isempty(tok), nums(k) = str2double(tok{1}{1}); end
    end
    [~, idx]  = max(nums);
    agentFile = fullfile(basePath, files(idx).name);
end

function obs = computeObs(state, pos_line, dir_line, V_trim)
% Returns raw (UN-normalised) 4-element obs vector for plotting and SLC
% comparison.  Do NOT feed this output directly to the agent — the env's
% step() and reset() handle normalisation internally.
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
