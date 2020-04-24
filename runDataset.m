% FilePath to the dataset and  ground truth labels and you need to change
% the X and Y variable below(look at the line number 28-30)
datasetFname = 'dataset/SalinasA_corrected.mat';
datasetGt = 'dataset/SalinasA_gt.mat';
datasetOut = 'SalinasA';

% train and test split - if you want to use 80-20 split(80% training and 20%
% testing), assign testSize=0.2, if you want 30-70 split(30% training and
% 70% tesing) assign testSize=0.7
testSize = 0.2;
% Number of datasets to be created
% numData = 10;

% Tensor decompostion rank
%ranks = [10, 50, 100, 200, 300, 500, 1000, 2000];
% ranks = [1000, 2000];


%Testing -- Don't un-comment below lines: only for testing
numData = 1;
ranks = [10];

data = load(datasetFname);
gt = load(datasetGt);

% IMPORTANT: Change X and Y according to variable stored in the .mat file
X = data.salinasA_corrected;
Y =  gt.salinasA_gt;

for r=ranks
    for i=1:numData
        createDataset(X, Y, testSize, r, datasetOut);
    end
end