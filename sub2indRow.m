function indxs=sub2indRow(s, i, j)
m = s(1);
n = s(2);
% assert(size(i) == size(j));
indxs = (i-1)*n + j;
end