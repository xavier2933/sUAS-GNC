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
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
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
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'value')
];

criticNet = dlnetwork(layerGraph(criticNet));
critic = rlValueFunction(criticNet, obsInfo, ...
    'ObservationInputNames', 'obs');

%% PPO agent options
agentOpts = rlPPOAgentOptions( ...
    'SampleTime',               env.Ts, ...
    'ExperienceHorizon',        2048, ...   % was 512; larger horizon = fewer updates/episode at peak
    'ClipFactor',               0.1, ...    % was 0.2; tighter trust region
    'EntropyLossWeight',        0.01, ...   % back to 0.01; 0.02 added variance in high-reward regime
    'NumEpoch',                 3, ...      % was 5; fewer reuse passes per update
    'MiniBatchSize',            64, ...
    'DiscountFactor',           0.99, ...
    'AdvantageEstimateMethod',  'gae', ...
    'GAEFactor',                0.95, ...
    'ActorOptimizerOptions',    rlOptimizerOptions('LearnRate', 3e-5), ...
    'CriticOptimizerOptions',   rlOptimizerOptions('LearnRate', 1e-3));

% Warm-start from the best checkpoint of the interrupted run.
% getActor/getCritic extracts the learned weights; rlPPOAgent rebinds them
% with the agentOpts declared above so all hyperparameters actually apply.
data        = load('saved_agents/run_0415_2145/Agent1371.mat');  % ← update to your best checkpoint
savedActor  = getActor(data.saved_agent);
savedCritic = getCritic(data.saved_agent);
agent       = rlPPOAgent(savedActor, savedCritic, agentOpts);  % new opts, old weights

% --- Train from scratch (uncomment if you want a clean slate) ---
% agent = rlPPOAgent(actor, critic, agentOpts);

% Each training run gets its own subfolder so files never mix or overwrite.
runTag   = datestr(now, 'mmdd_HHMM');          % e.g. '0413_1945'
agentDir = fullfile('saved_agents', ['run_' runTag]);
mkdir(agentDir);

%% Training options
% SaveAgentValue lowered: smoothness penalty reduces achievable per-episode
% reward vs the old reward function, so 3600 would never trigger.
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',          4000, ...
    'MaxStepsPerEpisode',   env.MaxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue',    8500, ...
    'Verbose',              true, ...
    'Plots',                'training-progress', ...
    'SaveAgentCriteria',    'EpisodeReward', ...
    'SaveAgentValue',       2800, ...      % lowered from 3600 to match new reward scale
    'SaveAgentDirectory',   agentDir);

%% Train
trainingStats = train(agent, env, trainOpts);
save('ppo_slg_agent.mat', 'agent');