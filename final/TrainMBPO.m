%% TrainMBPO.m
% Model-Based Policy Optimization (MBPO) for the StraightLine guidance task.
%
% Architecture:
%   - Real env      : StraightLineEnv  (ode45-based aircraft physics)
%   - Base agent    : SAC              (off-policy, continuous actions)
%   - World model   : rlNeuralNetworkEnvironment
%       * Transition : (obs, action) -> next_obs   [learned FC net -- unknown dynamics]
%       * Reward     : (obs, action, next_obs) -> reward   [function handle -- exact known formula]
%       * Is-done    : (obs, action, next_obs) -> isDone   [function handle -- exact known condition]
%
% Why function handles for reward/is-done?
%   The reward function and termination condition are fully known (from
%   StraightLineEnv) so there is no reason to learn them.  Only the
%   aircraft transition dynamics (ode45 + autopilot) need to be approximated
%   by a neural network. This saves two networks, removes two optimizer
%   hyperparameters, and gives exact reward/done signals during imagined rollouts.
%
% Workflow:
%   1. Collect ~1000 real steps (burn-in) -> fill experience replay buffer
%   2. Train transition NN on buffered (obs, act, nextObs) tuples
%   3. Roll out K imagined steps through the NN (fast -- no ode45)
%   4. Blend real + synthetic data and update SAC actor/critics
%   5. Repeat -- world model improves and imagined rollouts get cheaper/better

clear; close all;
addpath('..');
addpath('C:\Users\xavie\MATLAB\Projects\5128\hw7');

%% 1.  Real environment
env = StraightLineEnv();
validateEnvironment(env);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
numObs  = obsInfo.Dimension(1);   % 6
numAct  = actInfo.Dimension(1);   % 3

%% 2.  SAC base agent
% SAC (Soft Actor-Critic) is the standard base for MBPO — it is off-policy
% so model-generated transitions can be freely mixed into the replay buffer.
% rlAgentInitializationOptions auto-builds the Gaussian actor and twin
% Q-critics (no need to hand-wire networks like in the PPO script).
initOpts = rlAgentInitializationOptions('NumHiddenUnit', 64);

sacOpts = rlSACAgentOptions( ...
    'SampleTime',             env.Ts, ...
    'DiscountFactor',         0.99, ...
    'MiniBatchSize',          256, ...
    'ExperienceBufferLength', 1e5, ...
    'TargetSmoothFactor',     5e-3, ...     % soft target update tau
    'NumWarmStartSteps',      1000, ...     % random actions before SAC updates start
    'ActorOptimizerOptions',  rlOptimizerOptions('LearnRate', 3e-4), ...
    'CriticOptimizerOptions', rlOptimizerOptions('LearnRate', 3e-4));

baseAgent = rlSACAgent(obsInfo, actInfo, initOpts, sacOpts);

%% 3.  Transition model: (obs, action) -> next_obs
% This is the ONLY thing we need a neural network for — it approximates the
% aircraft EOM + autopilot dynamics without running ode45.
%
% Network topology:
%   obs (6)  -> FC(128) -\
%                         concat -> FC(256) -> ReLU -> FC(128) -> ReLU -> FC(6) -> next_obs
%   act (3)  -> FC(128) -/
transNet = buildTransitionNet(numObs, numAct);

transitionFcn = rlContinuousDeterministicTransitionFunction(transNet, obsInfo, actInfo, ...
    'ObservationInputNames',      'obsIn', ...
    'ActionInputNames',           'actIn', ...
    'NextObservationOutputNames', 'nextObsOut');

%% 4.  Assemble world model
% rewardFcn and isdoneFcn are function handles (defined at the bottom of
% this file) that reproduce the exact StraightLineEnv reward/termination.
worldModel = rlNeuralNetworkEnvironment(obsInfo, actInfo, ...
    transitionFcn, @mbpoReward, @mbpoIsDone);

%% 5.  MBPO agent
% Tuning rationale (aircraft dynamics are harder than they look):
%   - The SLC autopilot has internal state (integrators, mode logic) that
%     is NOT in the 6D observation -- making the transition non-Markovian
%     in obs-space and 1-step prediction harder than a pure physics sim.
%   - Rate terms (d_cross, d_course) are finite differences of consecutive
%     steps, adding temporal correlation a 1-step model struggles with.
%
% RealSampleRatio = 0.8  : 80% of each SAC mini-batch from REAL transitions.
%                          Lower toward 0.2 only once model loss stabilises.
% Horizon = 1            : single imagined step -- compounding error with
%                          hidden autopilot state drift makes >1 harmful.
% NumEpochForTrainingModel: more gradient passes per update so the model
%                          can keep pace with the expanding state coverage.
% LearnRate 3e-4         : lower than the original 1e-3 to avoid overshoot.

mbpoOpts = rlMBPOAgentOptions( ...
    'RealSampleRatio',            0.8, ...
    'NumEpochForTrainingModel',   5,   ...
    'TransitionOptimizerOptions', rlOptimizerOptions('LearnRate', 3e-4));

mbpoOpts.ModelRolloutOptions.Horizon = 1;   % single-step imagined rollouts

agent = rlMBPOAgent(baseAgent, worldModel, mbpoOpts);


%% 6.  Save directory (mirrors TrainPPO convention)
runTag   = datestr(now, 'mmdd_HHMM');
agentDir = fullfile('saved_agents', ['mbpo_' runTag]);
mkdir(agentDir);

%% 7.  Training options
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                2000, ...
    'MaxStepsPerEpisode',         env.MaxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          8500, ...
    'Verbose',                    true, ...
    'Plots',                      'training-progress', ...
    'SaveAgentCriteria',          'EpisodeReward', ...
    'SaveAgentValue',             2800, ...
    'SaveAgentDirectory',         agentDir);

%% 8.  Train
trainingStats = train(agent, env, trainOpts);
save('mbpo_slg_agent.mat', 'agent');


% =========================================================================
%  LOCAL FUNCTIONS  (must live at the bottom of the script file in MATLAB)
% =========================================================================

function net = buildTransitionNet(numObs, numAct)
% Build a two-input FC network: (obs, action) -> next_obs.
% Uses an additive merge (element-wise sum after projecting both inputs to
% the same width) — a common pattern for transition models.
    obsBranch = [
        featureInputLayer(numObs, 'Name', 'obsIn')
        fullyConnectedLayer(128,  'Name', 'obsFC')
    ];
    actBranch = [
        featureInputLayer(numAct, 'Name', 'actIn')
        fullyConnectedLayer(128,  'Name', 'actFC')
    ];
    trunk = [
        concatenationLayer(1, 2,  'Name', 'cat')
        fullyConnectedLayer(256,  'Name', 'fc1')
        reluLayer(                'Name', 'relu1')
        fullyConnectedLayer(128,  'Name', 'fc2')
        reluLayer(                'Name', 'relu2')
        fullyConnectedLayer(numObs,'Name','nextObsOut')
    ];
    lg = layerGraph(obsBranch);
    lg = addLayers(lg, actBranch);
    lg = addLayers(lg, trunk);
    lg = connectLayers(lg, 'obsFC', 'cat/in1');
    lg = connectLayers(lg, 'actFC', 'cat/in2');
    net = dlnetwork(lg);
end

% -------------------------------------------------------------------------
function reward = mbpoReward(obs, action, nextObs)
% Reproduces StraightLineEnv.computeReward for MBPO model rollouts.
% MBPO calls this in batched mode: nextObs{1} is numObs-x-N, so all
% arithmetic must operate column-wise and return a 1-x-N reward row.
%
% Smoothness penalty omitted -- world model doesn't track prev_action.
    no = nextObs{1};        % 6 x N  (N=batch size, usually 1 during rollout)
    cross_track = no(1,:); % 1 x N
    course_err  = no(2,:);
    h_err       = no(3,:);
    Va_err      = no(4,:);

    r_cross  = (cross_track / 100).^2;
    r_course = (course_err  / 0.5 ).^2;
    r_h      = (h_err       / 50  ).^2;
    r_va     = (Va_err      / 3   ).^2;

    reward = 1.0 - 0.4*(r_cross + r_course + r_h + r_va);  % 1 x N

    % Proximity bonus (vectorised)
    reward = reward + 1.0 * (abs(cross_track) < 5);        % 1 x N
end

% -------------------------------------------------------------------------
function isDone = mbpoIsDone(obs, action, nextObs)
% Termination condition from StraightLineEnv.
% Returns a 1-x-N logical row to match MBPO's batched indexing.
    no     = nextObs{1};          % 6 x N
    isDone = abs(no(1,:)) > 50;   % 1 x N logical
end
