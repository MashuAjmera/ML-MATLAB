clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

T= readtable('IRIS.csv');
[T.species,gN]=grp2idx(T.species);
n = size(T,1);
class=size(gN,1);

% Test train split parameters
idx = randperm(n);
ratio=0.9; % fraction of train set

% train set defined
X=T{idx(1:round(ratio*n)),1:4};
y=T{idx(1:round(ratio*n)),5};

% test set defined
X_test=T{idx(round(ratio*n)+1:end),1:4};
y_test=T{idx(round(ratio*n)+1:end),5};
test=size(X_test,1); % number of test instances
predict=zeros(test,1);

n_features = size(X,2);
lr = 0.001;
lambda_param = 0.01;

for i=1:(class-1)
    % ith train set
    Xi=X(y>=i,:);
    yi=y(y>=i);
    yi(yi==i)=1;
    yi(yi>i)=-1;
    n_samples=size(Xi,1); % number of training instances
    
    w = zeros(1,n_features);
    b = 0;
    
    for epoch=1:1000
        for idx=1:n_samples %for each training instance
            if yi(idx) * (dot(Xi(idx,:),w) + b) >= 1 % if classified correctly
                w =w- lr * (2 * lambda_param * w);
            else % if classified incorrectly
                w =w- lr * (2 * lambda_param * w - Xi(idx,:)*yi(idx));
                b =b- lr * yi(idx);
            end
        end
    end
    
    predict(predict<=0 & (X_test*w' + b)>0)=i; %assign 1 as this class
end
predict(predict==0)=class;

disp(predict');
disp(y_test');
error=sum(predict.*y_test<0);
accuracy=(1-error/test);
fprintf("The accuracy of the model is: %f",accuracy);