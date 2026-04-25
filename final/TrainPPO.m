%% TrainPPO.m
clear; close all;
addpath('..');
addpath('C:\Users\xavie\MATLAB\Projects\5128\hw7')
%% Create environment
env = StraightLineEnv();
validateEnvironment(env);   % sanity check — fix errors before training

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
numObs  = obsInfo.Dimension(1);   % 4
numAct  = actInfo.Dimension(1);   % 3

%% Build Actor network (outputs mean and std of Gaussian)
commonPath = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
];

meanPath = [
    fullyConnectedLayer(numAct, 'Name', 'meanFC')
    % No activation — rlContinuousGaussianActor rescales to action bounds
    % internally. tanh was saturating gradients near the action limits.
];

stdPath = [
    fullyConnectedLayer(numAct, 'Name', 'stdFC')
    softplusLayer('Name', 'std') % Ensures standard deviation is positive
];

% Assemble the graph
lgraph = layerGraph(commonPath);
lgraph = addLayers(lgraph, meanPath);
lgraph = addLayers(lgraph, stdPath);

% Connect the common backbone to both heads
lgraph = connectLayers(lgraph, 'relu2', 'meanFC');
lgraph = connectLayers(lgraph, 'relu2', 'stdFC');

actorNet = dlnetwork(lgraph);

% Update the Actor creation to point to BOTH outputs
actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    'ActionMeanOutputNames', 'meanFC', ...
    'ActionStandardDeviationOutputNames', 'std', ...
    'ObservationInputNames', 'obs');

%% Build Critic network (outputs scalar value)
criticNet = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'value')
];

criticNet = dlnetwork(layerGraph(criticNet));
critic = rlValueFunction(criticNet, obsInfo, ...
    'ObservationInputNames', 'obs');

%% PPO agent options
agentOpts = rlPPOAgentOptions( ...
    'SampleTime',               env.Ts, ...
    'ExperienceHorizon',        4096, ...  % spans ~1-2 full episodes per update for better diversity
    'ClipFactor',               0.1, ...    % was 0.2; tighter trust region
    'EntropyLossWeight',        0.01, ...   % back to 0.01; 0.02 added variance in high-reward regime
    'NumEpoch',                 3, ...      % was 5; fewer reuse passes per update
    'MiniBatchSize',            64, ...
    'DiscountFactor',           0.995, ...  % 0.99 (before) → 0.995: horizon 100→200 steps; needed for long recovery trajectories.
    'AdvantageEstimateMethod',  'gae', ...
    'GAEFactor',                0.95, ...
    'ActorOptimizerOptions',    rlOptimizerOptions('LearnRate', 3e-5), ...
    'CriticOptimizerOptions',   rlOptimizerOptions('LearnRate', 1e-3));

%% ── CONFIG ─────────────────────────────────────────────────────────────
% To resume from a checkpoint after a stage completes:
%   1. Stop the script (Ctrl-C) once the desired stage finishes
%   2. Set START_STAGE to the NEXT stage number (e.g. 2 after stage 1 finishes)
%   3. Set CHECKPOINT_FILE to the best Agent*.mat from the completed stage
%      e.g. 'saved_agents/run_0423_1045/stage1/Agent580.mat'
%   4. Re-run — the script will load the checkpoint and skip completed stages
START_STAGE      = 1;    % which curriculum stage to begin at (1 = fresh start)
%CHECKPOINT_FILE  = 'saved_agents/run_0423_1023/stage1/Agent497.mat';   % '' = build fresh agent; set path to resume from checkpoint
CHECKPOINT_FILE = '';
%% ────────────────────────────────────────────────────────────────────────

if isempty(CHECKPOINT_FILE)
    % Fresh agent — used on first run (START_STAGE = 1)
    agent = rlPPOAgent(actor, critic, agentOpts);
else
    % Resume from checkpoint — network architecture must match
    fprintf('Loading checkpoint: %s\n', CHECKPOINT_FILE);
    ckpt = load(CHECKPOINT_FILE);
    if isfield(ckpt, 'saved_agent'), agent = ckpt.saved_agent;
    elseif isfield(ckpt, 'agent'),   agent = ckpt.agent;
    else, f = fieldnames(ckpt); agent = ckpt.(f{1}); end
end

% Each training run gets its own subfolder so files never mix or overwrite.
runTag   = datestr(now, 'mmdd_HHMM');
agentDir = fullfile('saved_agents', ['run_' runTag]);
mkdir(agentDir);

%% Staged Curriculum Training
% Difficulty controls IC spread in StraightLineEnv.reset():
%   sig_pos = 15 + 120*difficulty  (m)
%   sig_chi = 0.15 + 0.25*difficulty (rad)
%
% Each stage runs until MaxEpisodes OR StopTrainingValue is hit,
% then advances to the next difficulty level.
difficulties   = [0.8];
maxEpsPerStage = [ 1400];

for s = START_STAGE:numel(difficulties)
    env.difficulty = difficulties(s);
    sig_pos_display = 15 + 120*difficulties(s);
    fprintf('\n=== Curriculum Stage %d/%d  |  difficulty=%.2f  |  sig_pos≈15+120*d=%.0fm ===\n', ...
        s, numel(difficulties), difficulties(s), sig_pos_display);

    stageDir = fullfile(agentDir, sprintf('stage%d', s));
    mkdir(stageDir);

    trainOpts = rlTrainingOptions( ...
        'MaxEpisodes',                maxEpsPerStage(s), ...
        'MaxStepsPerEpisode',         env.MaxSteps, ...
        'ScoreAveragingWindowLength', 20, ...
        'StopTrainingCriteria',       'AverageReward', ...
        'StopTrainingValue',          5500, ...  % 92% of theoretical max (6000); signals stage mastery
        'Verbose',                    true, ...
        'Plots',                      'training-progress', ...
        'SaveAgentCriteria',          'EpisodeReward', ...
        'SaveAgentValue',             4500, ...
        'SaveAgentDirectory',         stageDir);

    train(agent, env, trainOpts);   % agent is a handle object; mutates in place
end

save('ppo_slg_agent.mat', 'agent');
fprintf('\nTraining complete. Final agent saved to ppo_slg_agent.mat\n');