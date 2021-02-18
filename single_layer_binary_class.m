clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

% training to get Weight
input=[0 0 1; 0 1 1; 1 0 1; 1 1 1;];
correct_Output=[0 0 1 1];
Weight=2*rand(1,size(input,2))-1; % to get 3 positive as well as negative random numbers in one row 
for epoch=1:1000
    Weight=SGD_method(Weight,input,correct_Output);
end

% testing using the Weight obtained from training
test_input=[0 0 1; 0 1 1; 1 0 1; 1 1 1;];
N=length(test_input);
for k=1:N
    transposed_Input=test_input(k,:)';
    weighted_Sum=Weight*transposed_Input;
    output=Sigmoid(weighted_Sum)
end

% using Stochastic Gradient Descend as the Learning Rule
function Weight=SGD_method(Weight,input,correct_Output)
    alpha=0.9; % setting learning rate as 0.9
    N=length(input);
    for k=1:N
        % learning using forward propagation
        transposed_Input=input(k,:)';
        d=correct_Output(k);
        weighted_Sum=Weight*transposed_Input;
        output=Sigmoid(weighted_Sum);
        
        error=d-output;
        delta=output*(1-output)*error;
        dWeight=alpha*delta*transposed_Input;
        
        % adjusting weights using the new learned values
        for i=1:length(transposed_Input)
            Weight(i)=Weight(i)+dWeight(i);
        end
    end
end

% Sigmoid Function
function y=Sigmoid(x)
    y=1/(1+exp(-x));
end