
%%% Note: run the 'GenerateData_model_64_25_Res_Bnorm_Adam.m' to generate
%%% training data first.
clear;clc;
disp('start trainning....');
addpath('matconvnet-1.0-beta24\matconvnet-1.0-beta24\matlab');
vl_setupnn();

%%%-------------------------------------------------------------------------
%%% configuration
%%%-------------------------------------------------------------------------
opts.modelName        = 'model_25_Res_Bnorm_Dalited_Adam'; %%% model name
opts.learningRate     = [logspace(-3,-3,30) logspace(-4,-4,20) logspace(-4,-5,10)];%%% you can change the learning rate
opts.batchSize        = 16; %%% default
opts.gpus             = [1]; %%% this code can only support one GPU!

%%% solver
opts.solver           = 'Adam';

opts.gradientClipping = false; %%% Set 'true' to prevent exploding gradients in the beginning.
opts.expDir      = fullfile('data', opts.modelName);
opts.imdbPath    = fullfile(opts.expDir, 'imdb.mat');

%%%-------------------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
net  = feval(['heyi_DnCNN_init_',opts.modelName]);

%%%  load data into memory
imdb = load(opts.imdbPath) ;

%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------

[net, info] = DnCNN_train(net, imdb, ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






