clear, clc, close all
addpath('..');

x0 = zeros(12,1)
x0(7) = 69;
x0(6) = deg2rad(10);

res = utils.FlightPathAnglesFromState(x0)