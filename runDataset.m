% FilePath to the dataset and  ground truth labels and you need to change
% the X and Y variable below(look at the line number 8-10)
datasetFname = 'dataset/Indian_pines_corrected.mat';
datasetGt = 'dataset/Indian_pines_gt.mat';
outFile = 'IndianPines'; % Update this name for different datasets accordingly.

datasetOut = 'tensorDataset';

data = load(datasetFname);
gt = load(datasetGt);

% IMPORTANT: Change X and Y according to variable stored in the .mat(dataset) file
% X = data.salinasA_corrected;
% Y =  gt.salinasA_gt;
X = data.indian_pines_corrected;
Y = gt. indian_pines_gt;
folderPath = strcat(datasetOut, '/', outFile);
outFname = strcat(folderPath, '/', outFile);
if ~exist(folderPath, 'dir')
    disp(folderPath);
    mkdir(convertStringsToChars(folderPath));
end




% train and test split - if you want to use 80-20 split(80% training and 20%
% testing), assign testSize=0.2, if you want 30-70 split(30% training and
% 70% tesing) assign testSize=0.7
testSize = 0.7;

% Number of datasets to be created
% numData = 10;

% Tensor decompostion rank
%ranks = [10, 50, 100, 200, 300, 500, 1000, 2000];
% ranks = [1000, 2000];

%Testing -- Don't un-comment below lines: only for testing
numData = 2;
ranks = [10 20];



for r=ranks
    for i=1:numData
        createDataset(X, Y, testSize, r, outFname);
    end
end