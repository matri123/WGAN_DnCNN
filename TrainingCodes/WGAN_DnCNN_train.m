function [net, state] = WGAN_DnCNN_train(Gnet, Dnet, imdb, varargin)
addpath('./data/utilities');
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
opts.gpus = [1];
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
% Dnet.layers{end-1}.precious = 1;
% vl_simplenn_display(Gnet, 'batchSize', opts.batchSize) ;

state.getBatch = getBatch ;

%%%-------------------------------------------------------------------------
%%%  Train and Test
%%%-------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-epoch-%d.mat'], ep));
DmodelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'D-epoch-%d.mat'], ep));

start = findLastCheckpoint(opts.expDir,opts.modelName) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d', mfilename, start) ;
    load(modelPath(start), 'Gnet') ;
    load(DmodelPath(start), 'Dnet') ;
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
        Gnet = vl_simplenn_move(Gnet, 'gpu') ;
        Dnet = vl_simplenn_move(Dnet, 'gpu');
    end
    

    subset = state.train ;
    num = 0 ;
    res = [];
    
    batchStartTest = epoch;
    batchEndTest = min(epoch + opts.batchSize, numel(subset));
    batchTest = subset(batchStartTest : 1: batchEndTest);
    [blurTest, sharpTest] = state.getBatch(imdb, batchTest) ;
    
    resD = [];
    for t = 1 : opts.batchSize : numel(subset) / 3;
    %for t=1:opts.batchSize:opts.batchSize*5
        %%% get this image batch
        disp(strcat(num2str(epoch),'+', num2str(t)));
        batchStart = t;
        batchEnd = min(t + opts.batchSize - 1, numel(subset));
        batch = subset(batchStart : 1: batchEnd);
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        [blur, sharp] = state.getBatch(imdb, batch) ;
        if numel(opts.gpus) == 1
            blur = gpuArray(blur);
            sharp = gpuArray(sharp);
        end
        
%         dzdy = single(1);
%         Gnet.layers{end}.class = sharp;
%         resD = vl_simplenn(Gnet, blur, dzdy, resD, ...
%         'mode', 'normal', ...
%         'conserveMemory', opts.conserveMemory, ...
%         'backPropDepth', opts.backPropDepth, ...
%         'cudnn', opts.cudnn) ;
% 
%         for l = numel(Gnet.layers) : -1 : 1
%             disp(size(res(l).dzdw));
%             for j = 1 : numel(resD(l).dzdw)
% %                 disp(resD(l).dzdw{j});
%                 thisLR = state.learningRate * Gnet.layers{l}.learningRate(j);
%     %             disp(res(l).dzdw{j});
%                 Gnet.layers{l}.weights{j} = Gnet.layers{l}.weights{j} - thisLR * (1 / 35) * (resD(l).dzdw{j});
%             end
%         end
        
        
        
        % 1. G genarat deblur
        deblur = Ggenarate(Gnet, blur);
%         imshow(cat(2, im2uint8(blur(:,:,:,1)), im2uint8(deblur(:,:,:,1))));

        % 2. train D
        [Dnet, resD] = trainD(Dnet, sharp, deblur, resD, opts, state, opts.batchSize);

        % 3. train G
        Gnet = trainG(Gnet, Dnet, blur, opts, state, opts.batchSize);
    end

    
    %plot the error figure
    if numel(opts.gpus) == 1
        blurTest = gpuArray(blurTest);
        sharpTest = gpuArray(sharpTest);
    end
    res = vl_simplenn(Gnet, blurTest, [], [], 'conserveMemory', true, 'mode', 'test');
    deblurTest = blurTest - res(end).x;
    sumTest = 0;
    for i = 1 : size(blurTest, 4)
        [psnr, aaa] = Cal_PSNRSSIM(deblurTest(:,:,:,1), sharpTest(:,:,:,1), 0, 0);
        sumTest = sumTest + psnr;
    end
    sumTest = sumTest / size(blurTest, 4);
    TestError(epoch) = gather(sumTest);
    
    figure(1);clf;
%     hold on;
%     subplot(1,2,1);
%     plot(start+1:epoch,TrainError(start+1:epoch));
%     title('Training Error(all batch)')
%     xlabel('epoch');
%     ylabel('error');
    
    hold on;
    subplot(1,2,2);
    plot(start+1:epoch,TestError(start+1:epoch));
    title('Testing Error');
    xlabel('epoch');
    ylabel('error');
    
    drawnow;
    
    Gnet = vl_simplenn_move(Gnet, 'cpu');
    Dnet = vl_simplenn_move(Dnet, 'cpu');
    %%% save current model
    disp(strcat('saving model of epoch :',num2str(epoch),'......'));
    save(modelPath(epoch), 'Gnet');
    save(DmodelPath(epoch), 'Dnet')
    save('TestError.mat','TestError');
    disp('success')
    
end
end

%%%-------------------------------------------------------------------------
function [deblur] = Ggenarate(net, blur)
%%%-------------------------------------------------------------------------
    res = vl_simplenn(net, blur, [], [], 'conserveMemory', true, 'mode', 'normal');
    deblur = blur - res(end).x;
end

%%%-------------------------------------------------------------------------
function [Dnet, resD] = trainD(Dnet, sharp, deblur, resD, opts, state, batchSize)
%%%-------------------------------------------------------------------------
    inputs = [];
    labels = [];
    if numel(opts.gpus) == 1
        sharp1 = gather(sharp);
        deblur1 = gather(deblur);
    end
    for i = 1 : size(sharp1, 4)
        inputs(:,:,:,2 * i - 1) = sharp1(:,:,:,i);
        labels(2 * i - 1) = 1;
        inputs(:,:,:,2 * i) = deblur1(:,:,:,i);
        labels(2 * i) = 0;
    end

    if numel(opts.gpus) == 1
        inputs = single(inputs);
        labels = single(labels);
        inputs = gpuArray(inputs);
        labels = gpuArray(labels);
    end
    dzdy = single(1);
    
%     disp(Dnet.layers{1}.weights{1}(:,:,:,1));

    Dnet.layers{end}.class = labels;
    resD = vl_simplenn(Dnet, inputs, dzdy, resD, ...
        'mode', 'normal', ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'cudnn', opts.cudnn) ;
    
    for l = numel(Dnet.layers) : -1 : 1
%         disp(size(resD(l).dzdw));
        for j = 1 : numel(resD(l).dzdw)
%             disp(resD(l).dzdw{j});
%             disp(j);
            thisLR = state.learningRate * Dnet.layers{l}.learningRate(j);
%             disp(thisLR * (1 / batchSize) * (resD(l).dzdw{j}));
            Dnet.layers{l}.weights{j} = Dnet.layers{l}.weights{j} - thisLR * (1 / batchSize) * (resD(l).dzdw{j});
%             disp(Dnet.layers{l}.weights{j});
%         pause(10);
        end
    end
    
end

%%%-------------------------------------------------------------------------
function [Gnet] = trainG(Gnet, Dnet, blur, opts, state, batchSize)
%%%-------------------------------------------------------------------------
    netContainer.layers = {};
    for i = 1 : numel(Gnet.layers) 
        netContainer.layers{end + 1} = Gnet.layers{i};
    end
    for i = 1 : numel(Dnet.layers)
        netContainer.layers{end + 1} = Dnet.layers{i};
    end
    netContainer.layers{end}.loss = 'gloss';
%     disp('before');
%     disp(Gnet.layers{1}.weights{1}(:,:,:,1));

    dzdy = single(1);
    res = [];
    labels = ones(size(blur, 4), 1);
    netContainer.layers{end}.class = labels;
%     vl_simplenn_display(netContainer);
    if numel(opts.gpus) == 1
        netContainer = vl_simplenn_move(netContainer, 'gpu');
    end
    res = vl_simplenn(netContainer, blur, dzdy, res, ...
        'mode', 'normal', ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'cudnn', opts.cudnn);
    for l = numel(Gnet.layers) : -1 : 1
%         disp(l);
%         disp(numel(res(l).dzdw));
        for j = 1 : numel(res(l).dzdw)
%             disp(res(l).dzdw{j});
            thisLR = state.learningRate * netContainer.layers{l}.learningRate(j);
%             disp(thisLR * (1 / batchSize) * (res(l).dzdw{j}));
%             disp(netContainer.layers{l}.weights{j}(:,:,:,1));
            netContainer.layers{l}.weights{j} = netContainer.layers{l}.weights{j} - thisLR * (1 / batchSize) * (res(l).dzdw{j});
%             disp(netContainer.layers{l}.weights{j}(:,:,:,1));
%             pause(10);
        end
    end
%     disp('Con');
%     disp(netContainer.layers{1}.weights{1}(:,:,:,1));
%     disp('after');
%     disp(Gnet.layers{1}.weights{1}(:,:,:,1));
%     pause(2);
    
    for i = 1 : numel(Gnet.layers)
        Gnet.layers{i} = netContainer.layers{i};
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
end

%%%-------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir,modelName)
%%%-------------------------------------------------------------------------
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end

%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = theta;
A(A<-theta) = -theta;
end

%%%-------------------------------------------------------------------------
function fn = getBatch
%%%-------------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y);
end

%%%-------------------------------------------------------------------------
function [inputs,labels] = getSimpleNNBatch(imdb, batch)
%%%-------------------------------------------------------------------------
inputs = imdb.inputs(:,:,:,batch);
labels = imdb.labels(:,:,:,batch);
end


