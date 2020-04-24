    
function createDataset(X, Y, testSize, rank, outputFile)
X = tensor(double(X));


[i, j] = find(Y);
nonY = Y(find(Y));
% nonZ = size(i,1);
newX = X;
% m = ceil(0.2 * nonZ);
cv = cvpartition(nonY, 'HoldOut', testSize);
% missIds = datasample(s, 1:nonZ, m, 'Replace', false);
testI = i(cv.test);
testJ = j(cv.test);
trainI = i(cv.training);
trainJ = j(cv.training);
P = tenones(size(X));
zeroV(1:size(X,3)) = 0;
m = length(testI);
% newI = zeros(m,1);
% newJ = zeros(m,1);
for k=1:m
    P(testI(k), testJ(k), :) = zeroV;
    newX(testI(k), testJ(k), :) = zeroV;
end

[M,U,output] = cp_wopt(newX, P, rank);
err = relativeError(X, tensor(M));
rmse = calRMSE(X, tensor(M), testI, testJ);

A = M.U{1};
B = M.U{2};
C = M.U{3};
lambda = M.lambda;
%data = khatrirao(A,B);
timeNow = int32(posixtime(datetime('now', 'TimeZone', 'America/Los_Angeles')));
fname = strcat(outputFile, '_', num2str(rank), '_', num2str(timeNow), '.mat');
save(fname, 'A', 'B', 'C', 'lambda', 'Y', 'testI', 'testJ', 'trainI', 'trainJ', 'err', 'rmse', 'rank');
end

