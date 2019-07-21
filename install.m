% Compile libraries and download network

[home_dir, name, ext] = fileparts(mfilename('fullpath'));
% donwload network
    cd model/
    if ~(exist('imagenet-vgg-m-2048.mat', 'file') == 2)
        disp('Downloading the network "imagenet-vgg-m-2048.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat"...')
        urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat', 'imagenet-vgg-m-2048.mat')
        disp('Done!')
    end
    if ~(exist('imagenet-vgg-verydeep-16.mat', 'file') == 2)
        disp('Downloading the network "imagenet-vgg-verydeep-16.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"...')
        urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', 'imagenet-vgg-verydeep-16.mat')
        disp('Done!')
    end
    cd(home_dir)
% compile matconvnet
    setup_path();
    vl_setupnn();
    try
        disp('Trying to compile MatConvNet with GPU support')
        vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc')
    catch err
        disp('Installation failed!')
        warning('ASRCF:install', 'Could not compile MatConvNet with GPU support.\nVisit http://www.vlfeat.org/matconvnet/install/ for instructions of how to compile MatConvNet.\n');
    end
    status = movefile('mex/vl_*.mex*');
    cd(home_dir)