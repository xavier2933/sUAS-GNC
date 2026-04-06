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

%% Build Actor network (outputs mean of Gaussian)
actorNet = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numAct, 'Name', 'mean')
    tanhLayer('Name', 'tanh_out')   % squash to [-1,1], rescale in env
];

actorNet = dlnetwork(layerGraph(actorNet));
actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    'ActionMeanOutputNames', 'tanh_out', ...
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
    'ExperienceHorizon',        512, ...    % steps before each update
    'ClipFactor',               0.2, ...    % PPO clip epsilon
    'EntropyLossWeight',        0.01, ...   % encourage exploration
    'NumEpoch',                 10, ...     % SGD passes per update
    'MiniBatchSize',            64, ...
    'DiscountFactor',           0.99, ...
    'AdvantageEstimateMethod',  'gae', ...
    'GAEFactor',                0.95, ...
    'ActorOptimizerOptions',    rlOptimizerOptions('LearnRate', 3e-4), ...
    'CriticOptimizerOptions',   rlOptimizerOptions('LearnRate', 1e-3));

agent = rlPPOAgent(actor, critic, agentOpts);

%% Training options
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',          2000, ...
    'MaxStepsPerEpisode',   env.MaxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue',    250, ...        % tune based on your reward scale
    'Verbose',              true, ...
    'Plots',                'training-progress', ...
    'SaveAgentCriteria',    'EpisodeReward', ...
    'SaveAgentValue',       200, ...
    'SaveAgentDirectory',   'saved_agents');

%% Train
trainingStats = train(agent, env, trainOpts);
save('ppo_slg_agent.mat', 'agent');