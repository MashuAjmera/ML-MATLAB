clear
clc
close all

load accidents
x = hwydata(:,14);
t = hwydata(:,4);

k=5; % number of folds
N=length(x); % size of dataset
indices = crossvalind('Kfold',t,k);
limit=5;
sse=zeros(1,limit);
for n=1:limit % complexity of model
    for i = 1:k
        test = (indices == i);
        train = ~test;
        X=[];
        for idx=0:n
            X=[X,x(train,:).^idx];
        end
        w=(X'*X)\X'*t(train,:);
        H=[];
        for idx=0:n
            H=[H,x(test).^idx];
        end
        sse(n) = sse(n) + sum((t(test) - H*w).^2);
    end
end
err = sse / N