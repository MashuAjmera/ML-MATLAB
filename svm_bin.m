clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

T= readtable('IRIS.csv');
n = 100; % size(T,1)
T.species=grp2idx(T.species);
T.species(T.species==2)=-1;

% Test train split parameters
idx = randperm(n);
ratio=0.9; % fraction of train set

% train set defined
X=T{idx(1:round(ratio*n)),1:4};
y=T{idx(1:round(ratio*n)),5};
n_samples=size(X,1); % number of training instances

% test set defined
X_test=T{idx(round(ratio*n)+1:end),1:4};
y_test=T{idx(round(ratio*n)+1:end),5};
test=size(X_test,1); % number of test instances

n_features = size(X,2);
lr = 0.001;
lambda_param = 0.01;
w = zeros(1,n_features);
b = 0;

for epoch=1:1000
    for idx=1:n_samples
        if y(idx) * (dot(X(idx,:),w) + b) >= 1
            w =w- lr * (2 * lambda_param * w);
        else
            w =w- lr * (2 * lambda_param * w - X(idx,:)*y(idx));
            b =b- lr * y(idx);
        end
    end
end

predict = X_test*w' + b;
error=sum(predict.*y_test<0);
accuracy=(1-error/test);
fprintf("The accuracy of the model is: %f",accuracy);