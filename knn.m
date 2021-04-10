clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

T= readtable('diabetes.csv');
k=5; % number of nearest neighbours

% Test train split parameters
n = size(T,1);
idx = randperm(n);
ratio=0.9; % fraction of train set

% train set defined
train_input=table2array(T(idx(1:round(ratio*n)),1:8));
train_output=table2array(T(idx(1:round(ratio*n)),9));
train=size(train_input,1); % number of training instances

% test set defined
test_input=table2array(T(idx(round(ratio*n)+1:end),1:8));
actual_output=table2array(T(idx(round(ratio*n)+1:end),9));
test=size(test_input,1); % number of test instances

% testing each case
error=0.0;
predicted_output=zeros(1,test);
for i=1:test
    % distance from each train_input
    distance=zeros(1,train);
    class=zeros(1,train);
    for row=1:train
        % euclidean distance of each attribute
        e=0;
        for attribute=1:size(test_input,2)
            e=e+(test_input(i,attribute)-train_input(row,attribute))^2;
        end
        distance(row)=sqrt(e);
        class(row)=train_output(row);
    end
    
    % sorting the neighbours based on their distance
    temp=0;
    gemp=0;
    for l=1:length(distance)
        for j=1:(length(distance)-l)
            if(distance(j)>distance(j+1))
                temp=distance(j);
                distance(j)=distance(j+1);
                distance(j+1)=temp;
                gemp=class(j);
                class(j)=class(j+1);
                class(j+1)=gemp;
            end
        end
    end
    
    % counting the number of nearest neighbours with class 1 
    count=0;
    for j=1:k
        count=count+class(j);
    end
    predicted_output(i)=count>k-count;
    error=error+double(predicted_output(i)~=actual_output(i));  
end

accuracy=1-error/test;
fprintf('The total number of wrongly predicted cases is %d out of %d. So, the accuracy is %d.',error,test,accuracy);