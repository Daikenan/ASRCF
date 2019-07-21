function feat = get_vggmfeatures(im, use_sz, layers)

global vggmnet
global enableGPU

if isempty(vggmnet)
    initial_vggmnet();
end

sz_window = use_sz;
% Preprocessing
img = single(im);        % note: [0, 255] range
%img = imresize(img, vggmnet.meta.normalization.imageSize(1:2),'bicubic');
img = imResample(img, vggmnet.meta.normalization.imageSize(1:2));

average=vggmnet.meta.normalization.averageImage;

if numel(average)==3
    average=reshape(average,1,1,3);
end

img = bsxfun(@minus, img, average);

if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(vggmnet,img);

% Initialize feature maps
feat = cell(length(layers), 1);

for ii = 1:length(layers)
    
    % Resize to sz_window
    if enableGPU
        x = res(layers(ii)).x; 
   %     x= gather(x);
 %       x = imresize(x, sz_window(1:2));
    else
        x = res(layers(ii)).x;
    end
    x = imresize(x, sz_window(1:2),'bicubic');
  %  x = imResample(x, sz_window(1:2));
    
    % windowing technique
    
    feat{ii}=x;
    
end
feat=feat{1};

end
function initial_vggmnet()
% INITIAL_NET: Loading VGG-Net-16

global vggmnet;
vggmnet = load('./model/imagenet-vgg-m-2048.mat');

% Remove the fully connected layers and classification layer
vggmnet.layers(14+1:end) = [];

% Switch to GPU mode
global enableGPU;
if enableGPU
    vggmnet = vl_simplenn_move(vggmnet, 'gpu');
end

vggmnet=vl_simplenn_tidy(vggmnet);

end
