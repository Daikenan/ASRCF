% This is the demo script for ASRCF

clear;
clc;
close all;
setup_path();

% get seq information
base_path  = './seq';
video_path = [base_path '/' 'FaceOcc1'];
[seq, ground_truth] = load_video_info(video_path,'FaceOcc1');
seq.startFrame = 1;
seq.endFrame = seq.len;
seq.ground_truth=ground_truth;

% Run ASRCF- main function
results = run_ASRCF(seq, video_path);
