function net = init_discriminator

net.layers = {};

filtsize = 5;
pad = (filtsize - 1) / 2;
str = 1;
negval = 0.2;
lr  = [1 1] ;
weightDecay = [1 0];

%layer 1
% net.layers{end + 1} = struct( ...
%     'type', 'conv', ...
%     'weights', {{sqrt(1/9) * randn(5,5,1,32,'single'), zeros(1, 32, 'single')}}, ...
%     'stride', str, ...
%     'pad', pad, ...
%     'dilate', 1, ...
%     'learningRate', lr ...
% );
net.layers{end + 1} = initConvolution(filtsize, 1, 32, 1, 2);
net.layers{end + 1} = initReLU(negval);

%128
net.layers{end + 1} = initConvolution(filtsize, 32, 32, 2, 2);
net.layers{end + 1} = initReLU(negval);
net.layers{end + 1} = initConvolution(filtsize, 32, 64, 1, 2);
net.layers{end + 1} = initReLU(negval);

%64
net.layers{end + 1} = initConvolution(filtsize, 64, 64, 2, 2);
net.layers{end + 1} = initReLU(negval);
net.layers{end + 1} = initConvolution(filtsize, 64, 128, 1, 2);
net.layers{end + 1} = initReLU(negval);

%16
net.layers{end + 1} = initConvolution(filtsize, 128, 128, 4, 2);
net.layers{end + 1} = initReLU(negval);
net.layers{end + 1} = initConvolution(filtsize, 128, 256, 1, 2);
net.layers{end + 1} = initReLU(negval);

%4
net.layers{end + 1} = initConvolution(filtsize, 256, 256, 4, 2);
net.layers{end + 1} = initReLU(negval);
net.layers{end + 1} = initConvolution(filtsize, 256, 512, 1, 2);
net.layers{end + 1} = initReLU(negval);

%1
net.layers{end + 1} = initConvolution(4, 512, 512, 4, 0);
net.layers{end + 1} = initReLU(negval);
net.layers{end + 1} = initConvolution(1, 512, 1, 1, 0);
net.layers{end + 1} = struct('type', 'sigmoid');
end

function net = initConvolution(filtsize, instates, nStates, str, pad)
    lr  = [1 1] ;
    net = struct( ...
        'type', 'conv', ...
        'weights', {{sqrt(1/9) * randn(filtsize, filtsize, instates, nStates, 'single'), zeros(nStates, 1, 'single')}}, ...
        'stride', str, ...
        'pad', pad, ...
        'dilate', 1, ...
        'learningRate', lr, ...
        'opts', {{}} ...
    );
end

function net = initReLU(negval)
    net = struct( ...
        'type', 'relu', ...
        'leak', negval ...
    );
end

