
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
net1 = feval(['init_discriminator']);
vl_simplenn_display(net1);
for i = 1 : size(net1.layers,2)
    net1.layers{i}.precious = 1;
end

label = imread('../testsets/Set12/01.png');
label = im2single(label);
net.layers{end}.class = label;
res = vl_simplenn(net, label, [], [], 'conserveMemory', true, 'mode', 'test');
res1 = vl_simplenn(net1, label, [], [], 'conserveMemory', true, 'mode', 'test');
output = label - res(end).x;
disp(res1(end -1 ).x);

netsum.layers = {};
for i = 1 : size(net.layers, 2)
    netsum.layers{end + 1} = net.layers{i};
end
for i = 1 : size(net1.layers, 2)
    netsum.layers{end + 1} = net1.layers{i};
end
    
vl_simplenn_display(netsum);
% output = im2uint8(output);
% imshow(cat(2, im2uint8(label), im2uint8(output)));

%%%  load data into memory
%imdb = load(opts.imdbPath) ;

%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------

% [net, info] = DnCNN_train(net, imdb, ...
%     'expDir', opts.expDir, ...
%     'learningRate',opts.learningRate, ...
%     'solver',opts.solver, ...
%     'gradientClipping',opts.gradientClipping, ...
%     'batchSize', opts.batchSize, ...
%     'modelname', opts.modelName, ...
%     'gpus',opts.gpus) ;






