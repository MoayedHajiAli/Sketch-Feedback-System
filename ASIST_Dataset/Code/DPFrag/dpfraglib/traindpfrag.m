% traindpfrag.mat
%

%% Initialize
close all;
clear all;
clc;


%% Load params
% params = getparams('demo');
params = getparams('env');
params.setdir = 'PrimitiveAnnotation';
params.extension = '.mat';
params.douglasthresh = 3;

%% Load files
[trs, tes, trf, tef] = get_strokes(params.setdir, params.extension);

trs = [];
 for i = 1:length(trf)
 	trs = [trs read_sketch( trf{i} )];
 end

 %% Train
params = get_bestparams(trs, params); 
prec = dpseg_train(trs, params);

% save
save env;



