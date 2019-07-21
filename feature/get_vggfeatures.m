% GET_FEATURES: Extracting hierachical convolutional features

function feat = get_vggfeatures(im, use_sz, layers)

global vgg16net
global enableGPU
if isempty(vgg16net)
    initial_vgg16net();
end
sz_window = use_sz;

% Preprocessing
img = single(im);        % note: [0, 255] range
%img = imresize(img, vgg16net.meta.normalization.imageSize(1:2),'bicubic');
img = imResample(img, vgg16net.meta.normalization.imageSize(1:2));

average=vgg16net.meta.normalization.averageImage;

if numel(average)==3
    average=reshape(average,1,1,3);
end

img = bsxfun(@minus, img, average);

%if enableGPU, img = gpuArray(img);vgg16net = vl_simplenn_move(vgg16net,'gpu'); end
if enableGPU  
        img = gpuArray(img);
   %   vgg16net = vl_simplenn_move(vgg16net, 'gpu');
end

% Run the CNN
res = vl_simplenn(vgg16net,img);

% Initialize feature maps
feat = cell(length(layers), 1);

for ii = 1:length(layers)

    % Resize to sz_window
    if enableGPU
        
        x = res(layers(ii)).x;
      %  x = gather(x);
%        x = imresize(x, sz_window(1:2));
    else
        x = res(layers(ii)).x;
    end

x = imresize(x, sz_window(1:2),'bicubic');
%    x = imResample(x, sz_window(1:2));
  
    % windowing technique
    
    feat{ii}=x;
end
feat=feat{1};

end
function initial_vgg16net()
% INITIAL_NET: Loading VGG-Net-16
global vgg16net;

vgg16net = load('./model/imagenet-vgg-verydeep-16.mat');

% Remove the fully connected layers and classification layer
vgg16net.layers(30+1:end) = [];

% Switch to GPU mode
global enableGPU;
if enableGPU     
%     params.gpus=[1 2];
%     prepareGPUs(params, 1) ;
%     params.parameterServer.method = 'mmap' ;
%     params.parameterServer.prefix = 'mcn' ;
%     numGpus = numel(params.gpus) ;
%     if numGpus >= 1
       vgg16net = vl_simplenn_move(vgg16net, 'gpu');
%     end
%     if numGpus > 1
%       parserv = ParameterServer(params.parameterServer) ;
%       vl_simplenn_start_parserv(vgg16net, parserv) ;
%     else
%       parserv = [] ;
%     end
end

vgg16net=vl_simplenn_tidy(vgg16net);
end
% -------------------------------------------------------------------------
