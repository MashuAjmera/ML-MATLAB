clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

load accidents % Importing the dataset
x = hwydata(:,14); % taking the 14th column as input data
t = hwydata(:,4); % taking the 4th scolumn as output data

k=5; % number of folds
N=length(x); % size of dataset
indices = crossvalind('Kfold',t,k); % selecting the group number for each instance
limit=5; % taking maximum complexity limit
sse=zeros(1,limit); % initializing an array with zeroes for computing sse of each complexity

for n=1:limit % complexity of model
    for i = 1:k % for the given complexity performing k fold cross validation
        test = (indices == i); % taking the ith group for texting
        train = ~test; % and the remaining ones for training
        
        % to compute the matrix for powers of train input column wise
        X=[];
        for idx=0:n
            X=[X,x(train,:).^idx];
        end
        
        w=(X'*X)\X'*t(train,:); % using the result from least squares approximation for calculating w
        
        % to compute the matrix for powers of test input column wise
        H=[];
        for idx=0:n
            H=[H,x(test).^idx];
        end

        sse(n) = sse(n) + sum((t(test) - H*w).^2); %adding up the total sse for each complexity
    end
end

mse = sse / N % calculating the mean SSE
[minvalue, minidx] = min(mse); % function to calculate the model complexity with the least MSE
fprintf('The optimum model complexity is %d with an MSE of %f.', minidx, minvalue)