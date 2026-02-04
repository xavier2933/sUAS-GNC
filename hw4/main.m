clear, clc, close all
addpath('..');

params = utils.ttwistor();

vars = [20;0;200];



[Alon, Blon, Alat, Blat] = utils.Get_A_B_Lon_Lat(vars,params);

Alon

res = utils.RotationMatrix321([0, pi/3, pi/2])
res'

v_e_b = [10;-2;3];
v_a_b = [10;-1;-2];

res = v_e_b - v_a_b

res_e = utils.TransformFromBodyToInertial(res,[0, pi/3, pi/2])

