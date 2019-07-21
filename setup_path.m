function setup_path()

% Tracker implementation
addpath(genpath('feature/'));
addpath(genpath('implementation/'));

% Utilities
addpath('utils/');


% PDollar toolbox
addpath(genpath('external_libs/'));


vl_setupnn();
