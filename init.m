% This files creates initial folders required for code to run
% You only need to run this once. 
dataset = 'dataset';
results = 'results';
tensor = 'tensorDataset';
if ~exist(dataset, 'dir')
    disp(dataset);
    mkdir(convertStringsToChars(dataset));
end
if ~exist(results, 'dir')
    disp(results);
    mkdir(convertStringsToChars(results));
end
if ~exist(tensor, 'dir')
    disp(tensor);
    mkdir(convertStringsToChars(tensor));
end