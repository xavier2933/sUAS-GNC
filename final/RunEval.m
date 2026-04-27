%% RunEval.m
% Evaluation script to run multiple trials of PPO and SLC inference
% and collect performance metrics.

clear; close all;
addpath('..');
addpath('C:\Users\xavie\MATLAB\Projects\5128\hw7');

%% ─────────────────────────────────────────────
%%  CONFIG  
%% ─────────────────────────────────────────────
NUM_TRIALS    = 20;
AGENT_FILE    = 'saved_agents/run_0424_1942/stage1/Agent508.mat';
IC_DIFFICULTY = 1.0;  % 0=easy, 1=hard

% --- AIRCRAFT MASS OVERRIDE ---
OVERRIDE_MASS = false;
TEST_MASS_KG  = 4.5;   % [kg] New mass (nominal ttwistor is 5.74 kg)

% --- WIND OVERRIDE ---
INERTIAL_WIND  = [0; 0; 0]; % [North, East, Down] m/s steady wind (e.g., [0; 5; 0] for crosswind)

% Metric Thresholds
SETTLING_BOUNDARY = 5.0; % [m] boundary for settling time

%% ─────────────────────────────────────────────
%%  1. Load Agent
%% ─────────────────────────────────────────────
if isempty(AGENT_FILE)
    error('Please specify an AGENT_FILE to evaluate.');
end
fprintf('Loading agent from: %s\n', AGENT_FILE);
data = load(AGENT_FILE);
fnames = fieldnames(data);
if isfield(data, 'saved_agent')
    agent = data.saved_agent;
elseif isfield(data, 'agent')
    agent = data.agent;
else
    agent = data.(fnames{1});
end

%% ─────────────────────────────────────────────
%%  2. Environment Setup
%% ─────────────────────────────────────────────
env = StraightLineEnv();
env.difficulty = IC_DIFFICULTY;
env.wind_inertial = INERTIAL_WIND;

if OVERRIDE_MASS
    env.aircraft_parameters.m = TEST_MASS_KG;
    env.aircraft_parameters.W = TEST_MASS_KG * env.aircraft_parameters.g;
    trim_def = [env.V_trim; 0; 1805];
    [env.aircraft_state_trim, env.control_input_trim, ...
     env.trim_variables, ~] = utils.CalculateTrim(trim_def, env.aircraft_parameters);
end

rng('shuffle'); % random trials

Ts = env.Ts;
MaxSteps = env.MaxSteps;
pos_line = env.pos_line;
dir_line = env.dir_line;
V_trim   = env.V_trim;

% For SLC Autopilot setup
aircraft_parameters = utils.ttwistor();
if OVERRIDE_MASS
    aircraft_parameters.m = TEST_MASS_KG;
    aircraft_parameters.W = TEST_MASS_KG * aircraft_parameters.g;
end
kpath         = env.kpath;
chi_inf       = env.chi_inf;

load('ttwistor_gains_feed', '-mat');
control_gain_struct.Ts                = Ts;
control_gain_struct.takeoff_height   = -999;
control_gain_struct.height_hold_limit = 99999;


%% ─────────────────────────────────────────────
%%  3. Run Trials
%% ─────────────────────────────────────────────

% Storage for metrics
% PPO Metrics
rl_settling_times = NaN(NUM_TRIALS, 1);
rl_path_integrals = NaN(NUM_TRIALS, 1);
rl_smoothness     = NaN(NUM_TRIALS, 1);

% SLC Metrics
slc_settling_times = NaN(NUM_TRIALS, 1);
slc_path_integrals = NaN(NUM_TRIALS, 1);
slc_smoothness     = NaN(NUM_TRIALS, 1);

fprintf('\nStarting Evaluation over %d trials...\n', NUM_TRIALS);
fprintf('Difficulty: %.2f | Mass: %.2f kg | Wind: [%.1f, %.1f, %.1f]\n', ...
    IC_DIFFICULTY, env.aircraft_parameters.m, INERTIAL_WIND(1), INERTIAL_WIND(2), INERTIAL_WIND(3));
fprintf('------------------------------------------------------------\n');

for t = 1:NUM_TRIALS
    
    obs0 = reset(env);
    state0 = env.aircraft_state;
    
    % -- PPO Rollout --
    obs = obs0;
    isDone = false;
    k = 0;
    
    rl_raw_obs = zeros(4,  MaxSteps+1);
    rl_raw_obs(:,1) = computeObs(state0, pos_line, dir_line, V_trim);
    rl_action  = zeros(3,  MaxSteps);
    
    while ~isDone && k < MaxSteps
        action = cell2mat(getAction(agent, {obs}));
        [obs, ~, isDone, ~] = step(env, action);
        k = k + 1;
        rl_raw_obs(:,k+1) = computeObs(env.aircraft_state, pos_line, dir_line, V_trim);
        rl_action(:,k)    = action;
    end
    N_rl = k;
    rl_raw_obs = rl_raw_obs(:, 1:N_rl+1);
    rl_action  = rl_action(:, 1:N_rl);
    
    % Metrics - PPO
    ct_error = abs(rl_raw_obs(1,:));
    
    % 1. Settling Time: find last index where error > boundary
    % If it never exceeds, it's 0. If it NEVER settles, it's NaN.
    unsettled_idx = find(ct_error > SETTLING_BOUNDARY, 1, 'last');
    if isempty(unsettled_idx)
        rl_settling_times(t) = 0;
    elseif unsettled_idx == length(ct_error)
        rl_settling_times(t) = NaN; % Did not settle
    else
        rl_settling_times(t) = unsettled_idx * Ts;
    end
    
    % 2. Path Integral Error (sum(|e_y|) * dt)
    rl_path_integrals(t) = sum(ct_error) * Ts;
    
    % 3. Control Smoothness: Proxied by squared variations in PPO actions
    rl_smoothness(t) = sum(sum(diff(rl_action, 1, 2).^2)) / max(1, N_rl);
    
    
    % -- SLC Rollout --
    slc_state = zeros(12, MaxSteps+1);
    slc_state(:,1) = state0;
    N_slc = MaxSteps;
    slc_raw_obs = zeros(4, MaxSteps+1);
    slc_raw_obs(:,1) = rl_raw_obs(:,1);
    
    slc_actions = zeros(4, MaxSteps); % store controls for smoothness metric (e,a,r,t)
    
    for i = 1:MaxSteps
        TSPAN = Ts * [i-1, i];
        wind_body   = utils.TransformFromInertialToBody(INERTIAL_WIND, slc_state(4:6,i));
        air_rel     = slc_state(7:9,i) - wind_body;
        wind_angles = utils.AirRelativeVelocityVectorToWindAngles(air_rel);

        control_objectives = utils.StraightLineGuidance( ...
            pos_line, dir_line, slc_state(1:3,i), kpath, chi_inf, V_trim);

        [control_out, ~] = hw7utils.SLCWithFeedForwardAutopilot( ...
            Ts*(i-1), slc_state(:,i), wind_angles, control_objectives, control_gain_struct);
        
        slc_actions(:,i) = control_out(1:4);

        [~, YOUT] = ode45(@(t,y) utils.AircraftEOM( ...
            t, y, control_out, INERTIAL_WIND, aircraft_parameters), ...
            TSPAN, slc_state(:,i), []);

        slc_state(:,i+1) = YOUT(end,:)';
        slc_raw_obs(:,i+1) = computeObs(slc_state(:,i+1), pos_line, dir_line, V_trim);
        
        if abs(slc_raw_obs(1,i+1)) > 200 || -slc_state(3,i+1) < 50
            N_slc = i;
            break;
        end
    end
    slc_raw_obs = slc_raw_obs(:, 1:N_slc+1);
    slc_actions = slc_actions(:, 1:N_slc);
    
    % Metrics - SLC
    ct_error_slc = abs(slc_raw_obs(1,:));
    slc_crashed = (N_slc < MaxSteps);  % only true if the break fired
    
    unsettled_idx_slc = find(ct_error_slc > SETTLING_BOUNDARY, 1, 'last');
    if isempty(unsettled_idx_slc)
        slc_settling_times(t) = 0;
    elseif slc_crashed
        slc_settling_times(t) = NaN;  % genuinely crashed before settling
    elseif unsettled_idx_slc == length(ct_error_slc)
        slc_settling_times(t) = NaN;  % never settled even without crash
    else
        slc_settling_times(t) = unsettled_idx_slc * Ts;
    end
    
    slc_path_integrals(t) = sum(ct_error_slc) * Ts;
    
    % SLC Smoothness is proxied by control surface variations
    slc_smoothness(t) = sum(sum(diff(slc_actions, 1, 2).^2)) / max(1, N_slc);
    
    % Print single trial results
    % Note: NaN prints as NaN which is correct
    fprintf('Trial %2d | PPO -> Settling: %6.1fs | PIE: %6.1f | Smoothness: %.4f \n', ...
        t, rl_settling_times(t), rl_path_integrals(t), rl_smoothness(t));
end

%% ─────────────────────────────────────────────
%%  4. Final Report
%% ─────────────────────────────────────────────
fprintf('\n============================================================\n');
fprintf('EVALUATION RESULTS (Over %d trials)\n', NUM_TRIALS);
fprintf('============================================================\n');

% Helper to compute stats safely ignoring NaNs
compute_stats = @(x) [sum(~isnan(x)), mean(x, 'omitnan'), std(x, 'omitnan')];

rl_set_stats = compute_stats(rl_settling_times);
slc_set_stats = compute_stats(slc_settling_times);

rl_pie_stats = compute_stats(rl_path_integrals);
slc_pie_stats = compute_stats(slc_path_integrals);

rl_smt_stats = compute_stats(rl_smoothness);
slc_smt_stats = compute_stats(slc_smoothness);

fprintf('\n--- PPO AGENT ---\n');
fprintf('Settling Time (<%dm): Mean = %6.2fs | Std = %6.2fs  (%d/%d settled)\n', ...
    SETTLING_BOUNDARY, rl_set_stats(2), rl_set_stats(3), rl_set_stats(1), NUM_TRIALS);
fprintf('Path Integral Error:   Mean = %6.2f  | Std = %6.2f\n', ...
    rl_pie_stats(2), rl_pie_stats(3));
fprintf('Control Smoothness:    Mean = %6.4f  | Std = %6.4f\n', ...
    rl_smt_stats(2), rl_smt_stats(3));

fprintf('\n--- SLC AUTOPILOT ---\n');
fprintf('Settling Time (<%dm): Mean = %6.2fs | Std = %6.2fs  (%d/%d settled)\n', ...
    SETTLING_BOUNDARY, slc_set_stats(2), slc_set_stats(3), slc_set_stats(1), NUM_TRIALS);
fprintf('Path Integral Error:   Mean = %6.2f  | Std = %6.2f\n', ...
    slc_pie_stats(2), slc_pie_stats(3));
fprintf('Control Smoothness:    Mean = %6.4f  | Std = %6.4f\n', ...
    slc_smt_stats(2), slc_smt_stats(3));
fprintf('============================================================\n');


%% ─────────────────────────────────────────────
function obs = computeObs(state, pos_line, dir_line, V_trim)
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
