function [net, state] = WGAN_DnCNN_train(Gnet, Dnet, imdb, varargin)

%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%%%-------------------------------------------------------------------------
%%% solvers: SGD(default) and Adam with(default)/without gradientClipping
%%%-------------------------------------------------------------------------

%%% solver: Adam
%%% opts.solver = 'Adam';
opts.beta1   = 0.9;
opts.beta2   = 0.999;
opts.alpha   = 0.01;
opts.epsilon = 1e-8;


%%% solver: SGD
opts.solver = 'SGD';
opts.learningRate = 0.01;
opts.weightDecay  = 0.0001;
opts.momentum     = 0.9 ;

%%% GradientClipping
opts.gradientClipping = false;
opts.theta            = 0.005;

%%% specific parameter for Bnorm
opts.bnormLearningRate = 0;

%%%-------------------------------------------------------------------------
%%%  setting for simplenn
%%%-------------------------------------------------------------------------

opts.conserveMemory = false ;
opts.mode = 'normal';
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts.TestErrorFile='';
%%%-------------------------------------------------------------------------
%%%  setting for model
%%%-------------------------------------------------------------------------

opts.batchSize = 128 ;
opts.gpus = [];
opts.numEpochs = 200 ;
opts.modelName   = 'model';
opts.expDir = fullfile('data',opts.modelName) ;
opts.train = find(imdb.set==1);
opts.test  = find(imdb.set==2);

%%%-------------------------------------------------------------------------
%%%  update settings
%%%-------------------------------------------------------------------------

opts = vl_argparse(opts, varargin);
opts.numEpochs = numel(opts.learningRate);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%%%-------------------------------------------------------------------------
%%%  Initialization
%%%-------------------------------------------------------------------------

Gnet = vl_simplenn_tidy(Gnet);    %%% fill in some eventually missing values
Dnet = vl_simplenn_tidy(Dnet);
Gnet.layers{end-1}.precious = 1;
vl_simplenn_display(Gnet, 'batchSize', opts.batchSize) ;

state.getBatch = getBatch ;

%%%-------------------------------------------------------------------------
%%%  Train and Test
%%%-------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-epoch-%d.mat'], ep));

start = findLastCheckpoint(opts.expDir,opts.modelName) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d', mfilename, start) ;
    load(modelPath(start), 'net') ;
    % net = vl_simplenn_tidy(net) ;
end

%%% we store the test error during training
%%% check the TestErrorFile
if(~exist(opts.TestErrorFile,'file'))
    TestError=[];
else
    load(opts.TestErrorFile,'TestError')
end

TrainError=[];
for epoch = start+1 : opts.numEpochs 
    %%% Train for one epoch.
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
    state.train = opts.train(randperm(numel(opts.train))) ; %%% shuffle
    state.test  = opts.test; %%% no need to shuffle
    opts.thetaCurrent = opts.theta(min(epoch, numel(opts.theta)));
    if numel(opts.gpus) == 1
        net = vl_simplenn_move(net, 'gpu') ;
    end
    

    subset = state.train ;
    num = 0 ;
    res = [];
    for t = 1 : opts.batchSize : numel(subset) 
    %for t=1:opts.batchSize:opts.batchSize*5
        %%% get this image batch
        batchStart = t;
        batchEnd = min(t + opts.batchSize - 1, numel(subset));
        batch = subset(batchStart : 1: batchEnd);
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        [blur, sharp] = state.getBatch(imdb, batch) ;
        % 1. G genarat deblur
        deblur = Ggenarate(Gnet, blur);

        % 2. train D
        Dnet = trainD(Dnet, sharp, deblur, opts, state, opts.batchSize);

        % 3. train G
        [Gnet, state, TrainError(epoch)] = G_process_epoch(Gnet, state, imdb, opts, 'train');
        [Gnet,  ~  ,TestError(epoch)] = G_process_epoch(Gnet, state, imdb, opts, 'test' );


    
    %plot the error figure
    
    figure(1);clf;
    hold on;
    subplot(1,2,1);
    plot(start+1:epoch,TrainError(start+1:epoch));
    title('Training Error(all batch)')
    xlabel('epoch');
    ylabel('error');
    
    hold on;
    subplot(1,2,2);
    plot(start+1:epoch,TestError(start+1:epoch));
    title('Testing Error');
    xlabel('epoch');
    ylabel('error');
    
    drawnow;
    
    net = vl_simplenn_move(net, 'cpu');
    %%% save current model
    disp(strcat('saving model of epoch :',num2str(epoch),'......'));
    save(modelPath(epoch), 'net')
    save('TestError.mat','TestError');
    disp('success')
    
end

%%%-------------------------------------------------------------------------
function [deblur] = Ggenarate(net, blur)
%%%-------------------------------------------------------------------------
    for i = 1 : numel(blur)
        res = vl_simplenn(net, blur[i], [], [], 'conserveMemory', true, 'mode', 'test');
        deblur(i) = blur(i) - res{end}.x;
    end
end

%%%-------------------------------------------------------------------------
function [Dnet] = trainD(Dnet, sharp, deblur, opts, state, batchSize)
%%%-------------------------------------------------------------------------
    inputs = []
    labels = []
    for i = 1 : numel(sharp)
        inputs(i) = sharp(i);
        labels(i) = 1;
    end
    for i = 1 : numel(deblur)
        inputs(numel(sharp) + i) = deblur(i);
        labels(numel(sharp) + i) = 0;
    end
    randp = randperm(numel(sharp) + numel(deblur));
    inputs = inputs(randp);
    labels = labels(randp);

    Dnet.layers{end}.class = labels;
    res = vl_simplenn(Dnet, inputs, dzdy, res, ...
        'mode', 'normal', ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'cudnn', opts.cudnn) ;

    for l = numel(Dnet.layers) : -1 : 1
        for j = 1 : numel(res(l).dzdw)
            thisLR = state.learningRate * net.layers{l}.learningRate(j);
            net.layers{l}.weights{j} = net.layers{l}.weights{j} - thisLR * (1 / batchSize) * (res(l).dzdw{j});
        end
    end
end

%%%-------------------------------------------------------------------------
function [Gnet] = trainG(Gnet, Dnet, blur, opts, state, batchSize)
%%%-------------------------------------------------------------------------
    netContainer.layers = {};
    for i = 1 : numel(Gnet) 
        netContainer.layers{end + 1} = Gnet.layers{i};
    end
    for i = 1 : numel(Dnet)
        netContainer.layers{end + 1} = Dnet.layers{i};
    end

    labels = ones(numel(blur), 1);
    netContainer.layers{end}.class = labels;
    res = vl_simplenn(netContainer, blur, dzdy, res, ...
        'mode', 'normal', ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'cudnn', opts.cudnn);
    for l = numel(Gnet.layers) : -1 : 1
        for j = 1 : numel(res(l).dzdw)
            thisLR = state.learningRate * net.layers{l}.learningRate(j);
            net.layers{l},weights{j} = net.layers{l}.weights{j} - thisLR * (1 / batchSize) * (res(l).dzdw{j});
        end
    end
end


%%%-------------------------------------------------------------------------
function  [net, state,E] = process_epoch(net, state, imdb, opts, mode)
%%%-------------------------------------------------------------------------

if strcmp(mode,'train')
    
    switch opts.solver
        
        case 'SGD' %%% solver: SGD
            for i = 1:numel(net.layers)
                if isfield(net.layers{i}, 'weights')
                    for j = 1:numel(net.layers{i}.weights)
                        state.layers{i}.momentum{j} = 0;
                    end
                end
            end
            
        case 'Adam' %%% solver: Adam
            for i = 1:numel(net.layers)
                if isfield(net.layers{i}, 'weights')
                    for j = 1:numel(net.layers{i}.weights)
                        state.layers{i}.t{j} = 0;
                        state.layers{i}.m{j} = 0;
                        state.layers{i}.v{j} = 0;
                    end
                end
            end
            
    end
    
end


subset = state.(mode) ;
num = 0 ;
res = [];
E=0;
for t=1:opts.batchSize:numel(subset) 
%for t=1:opts.batchSize:opts.batchSize*5
    %%% get this image batch
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1, numel(subset));
    batch = subset(batchStart : 1: batchEnd);
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
    
    [inputs,labels] = state.getBatch(imdb, batch) ;
    
    if numel(opts.gpus) == 1
        inputs = gpuArray(inputs);
        labels = gpuArray(labels);
    end
    
    if strcmp(mode, 'train')
        dzdy = single(1);
        evalMode = 'normal';%%% forward and backward (Gradients)
    else
        dzdy = [] ;
        evalMode = 'test';  %%% forward only
    end
    
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, inputs, dzdy, res, ...
        'mode', evalMode, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'cudnn', opts.cudnn) ;
    
    if strcmp(mode, 'train')
        [state, net] = params_updates(state, net, res, opts, opts.batchSize) ;
    end
    
    lossL2 = gather(res(end).x) ;
    E=E+lossL2;
    
    %%%--------add your code here------------------------
    
    %%%--------------------------------------------------
    
    fprintf('%s: epoch %02d: %3d/%3d:', mode, state.epoch, ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    fprintf('error: %f \n', lossL2) ;
    
end


%%%-------------------------------------------------------------------------
function [state, net] = params_updates(state, net, res, opts, batchSize)
%%%-------------------------------------------------------------------------

switch opts.solver
    
    case 'SGD' %%% solver: SGD
        
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = ...
                        (1 - thisLR) * net.layers{l}.weights{j} + ...
                        (thisLR/batchSize) * res(l).dzdw{j} ;
                else
                    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j);
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/thisLR;
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * gradientClipping(res(l).dzdw{j},theta) ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    else
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * res(l).dzdw{j} ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    end
                end
            end
        end
        
        
    case 'Adam'  %%% solver: Adam
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = ...
                        (1 - thisLR) * net.layers{l}.weights{j} + ...
                        (thisLR/batchSize) * res(l).dzdw{j} ;
                else
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    state.layers{l}.t{j} = state.layers{l}.t{j} + 1;
                    t = state.layers{l}.t{j};
                    alpha = thisLR;
                    lr = alpha * sqrt(1 - opts.beta2^t) / (1 - opts.beta1^t);
                    
                    state.layers{l}.m{j} = state.layers{l}.m{j} + (1 - opts.beta1) .* (res(l).dzdw{j} - state.layers{l}.m{j});
                    state.layers{l}.v{j} = state.layers{l}.v{j} + (1 - opts.beta2) .* (res(l).dzdw{j} .* res(l).dzdw{j} - state.layers{l}.v{j});
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/lr;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * gradientClipping(state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon),theta);
                    else
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon);
                    end
                    
                end
            end
        end
end


%%%-------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir,modelName)
%%%-------------------------------------------------------------------------
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = theta;
A(A<-theta) = -theta;

%%%-------------------------------------------------------------------------
function fn = getBatch
%%%-------------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y);

%%%-------------------------------------------------------------------------
function [inputs,labels] = getSimpleNNBatch(imdb, batch)
%%%-------------------------------------------------------------------------
inputs = imdb.inputs(:,:,:,batch);
labels = imdb.labels(:,:,:,batch);



