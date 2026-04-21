%% TrainSAC.m
% Soft Actor-Critic (SAC) for the StraightLine guidance task.
%
% SAC vs PPO -- key differences:
%   PPO  : on-policy.  Collects a batch of transitions with the CURRENT
%          policy, does a few gradient steps, then throws all data away.
%          Data-hungry but stable.
%   SAC  : off-policy.  Every real (obs,act,rew,nextObs,done) tuple is
%          stored in a replay buffer and can be reused for many gradient
%          updates long after it was collected.  More sample-efficient.
%          Also adds an entropy bonus that naturally balances exploration.
%
% Neither SAC nor PPO does multi-step lookahead in the sim during training.
% Both learn purely from single-step interactions with StraightLineEnv.
%
% Workflow:
%   1.  Take one real step in StraightLineEnv (ode45 + autopilot)
%   2.  Store transition in replay buffer (size 1e5)
%   3.  After NumWarmStartSteps, sample a mini-batch and update
%       actor + twin Q-critics + entropy temperature
%   4.  Repeat until convergence

clear; close all;
addpath('..');
addpath('C:\Users\xavie\MATLAB\Projects\5128\hw7');

%% 1.  Environment  (unchanged from TrainPPO)
env = StraightLineEnv();
validateEnvironment(env);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% 2.  SAC agent
% rlAgentInitializationOptions auto-builds:
%   - Gaussian actor (same structure as PPO actor)
%   - Two Q-critics  (SAC's "clipped double-Q" trick for stability)
% NumHiddenUnit = 64 matches the 64-unit layers used in TrainPPO.
initOpts = rlAgentInitializationOptions('NumHiddenUnit', 64);

agentOpts = rlSACAgentOptions( ...
    'SampleTime',             env.Ts, ...
    'DiscountFactor',         0.99, ...
    'MiniBatchSize',          256, ...
    'ExperienceBufferLength', 1e5, ...
    'TargetSmoothFactor',     1e-3, ...         % slowed from 5e-3 -- less aggressive target updates
    'NumWarmStartSteps',      1000, ...
    'ActorOptimizerOptions',  rlOptimizerOptions('LearnRate', 1e-4), ...   % halved from 3e-4
    'CriticOptimizerOptions', rlOptimizerOptions('LearnRate', 1e-4));       % halved from 3e-4

% --- Warm-start from best checkpoint ------------------------------------
% Uncomment this block (and comment the scratch line below) to resume
% from a saved checkpoint with new hyperparameters.
%
% data      = load('saved_agents/sac_0420_1830/Agent192.mat');  % <-- update path
% agent     = data.saved_agent;
% agentOpts           = agent.AgentOptions;
% agentOpts.ActorOptimizerOptions.LearnRate     = 1e-4;
% agentOpts.CriticOptimizerOptions(1).LearnRate = 1e-4;
% agentOpts.CriticOptimizerOptions(2).LearnRate = 1e-4;
% agentOpts.TargetSmoothFactor                  = 1e-3;
% agent.AgentOptions = agentOpts;

% --- Scratch run --------------------------------------------------------
agent = rlSACAgent(obsInfo, actInfo, initOpts, agentOpts);




%% 3.  Save directory  (mirrors TrainPPO convention)
runTag   = datestr(now, 'mmdd_HHMM');
agentDir = fullfile('saved_agents', ['sac_' runTag]);
mkdir(agentDir);

%% 4.  Training options
% Reward scale with new function (3000 steps):
%   ~6000 = theoretical max (perfect tracking)
%   ~5000 = good  (~3m cross-track, ~3° course, ~2m height)
%   ~2500 = moderate (oscillating but on the line)
%   < 1000 = poor / crashing
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                2000, ...
    'MaxStepsPerEpisode',         env.MaxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          5000, ...      % good tracking (~3m cross-track avg)
    'Verbose',                    true, ...
    'Plots',                      'training-progress', ...
    'SaveAgentCriteria',          'EpisodeReward', ...
    'SaveAgentValue',             2200, ...      % saves anything in the moderate+ range
    'SaveAgentDirectory',         agentDir);


%% 5.  Train
trainingStats = train(agent, env, trainOpts);
save('sac_slg_agent.mat', 'agent');
