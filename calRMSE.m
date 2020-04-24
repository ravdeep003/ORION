function err = calRMSE(Xog, X, I, J)
m = size(I,1);
total = 0;
for k=1:m
    i = I(k);
    j = J(k);
    diff = double(Xog(i,j,:)) - double(X(i,j,:));
    total = total + sum(diff.^2);   
end
n = m * size(Xog,3);
display(n);
err = sqrt(total/n);
end
