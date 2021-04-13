clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

T= readtable('cancer.csv');
[T.Var2,gN]=grp2idx(T.Var2);
T.Var2=T.Var2-1;

% Test train split parameters
m = size(T,1);
idx = randperm(m);

% training to get Weight
input=table2array(T(idx(1:round(0.9*m)),3:32));
correct_Output=table2array(T(idx(1:round(0.9*m)),2));

w1=2*rand(20,size(input,2))-1; % input weights for first hidden layer with 20 neurons from 30 input layer neurons
w2=2*rand(20,20)-1; % input weights for second hidden layer with 20 neurons
w3=2*rand(20,20)-1; % input weights for third hidden layer with 20 neurons
w4=2*rand(1,20)-1; % input weights for output layer with 1 neuron since binary classification

for epoch=1:10000
    [w1,w2,w3,w4]=SGD_method(w1,w2,w3,w4,input,correct_Output);
end

% testing using the Weight obtained from training
test_input=table2array(T(idx(round(0.9*m)+1:end),3:32));
N=length(test_input);
final_output=strings(1,N);
for k=1:N
    test_input_Value=test_input(k,:)';
    
    input_of_hidden_layer1=w1*test_input_Value;
    output_of_hidden_layer1=ReLU(input_of_hidden_layer1);
    
    input_of_hidden_layer2=w2*output_of_hidden_layer1;
    output_of_hidden_layer2=ReLU(input_of_hidden_layer2);
    
    input_of_hidden_layer3=w3*output_of_hidden_layer2;
    output_of_hidden_layer3=ReLU(input_of_hidden_layer3);
    
    input_of_output_node=w4*output_of_hidden_layer3;
    
    if Sigmoid(input_of_output_node)<0.5
        final_output(k)=Sigmoid(input_of_output_node);
    else 
        final_output(k)=Sigmoid(input_of_output_node);
    end
end
disp(final_output);

% using Stochastic Gradient Descend as the Learning Rule
function [w1,w2,w3,w4]=SGD_method(w1,w2,w3,w4,input,correct_Output)
    alpha=0.4; % setting learning rate as 0.9
    N=length(input);
    for k=1:N        
        reshaped_input_Value=input(k,:)';
        
        % learning using forward propagation
        input_of_hidden_layer1=w1*reshaped_input_Value;
        output_of_hidden_layer1=ReLU(input_of_hidden_layer1);
        
        input_of_hidden_layer2=w2*output_of_hidden_layer1;
        output_of_hidden_layer2=ReLU(input_of_hidden_layer2);
        
        input_of_hidden_layer3=w3*output_of_hidden_layer2;
        output_of_hidden_layer3=ReLU(input_of_hidden_layer3);
        
        input_of_output_node=w4*output_of_hidden_layer3;
        final_output=Sigmoid(input_of_output_node);
        
        correct_Output_transpose=correct_Output(k);
        delta=correct_Output_transpose-final_output;
        
        % error back propagation to third hidden layer
        error_of_hidden_layer3=w4'*delta;
        delta3=(input_of_hidden_layer3>0).*error_of_hidden_layer3;
        
        % error back propagation to second hidden layer
        error_of_hidden_layer2=w3'*delta3;
        delta2=(input_of_hidden_layer2>0).*error_of_hidden_layer2;
        
        % error back propagation to first hidden layer
        error_of_hidden_layer1=w2'*delta2;
        delta1=(input_of_hidden_layer1>0).*error_of_hidden_layer1;
        
        % adjusting weights using the new learned values
        w1=w1+alpha*delta1*reshaped_input_Value';
        w2=w2+alpha*delta2*output_of_hidden_layer1';
        w3=w3+alpha*delta3*output_of_hidden_layer2';
        w4=w4+alpha*delta*output_of_hidden_layer3';
    end
end

function y=ReLU(x)
y=max(0,x);
end

% Sigmoid Function
function y=Sigmoid(x)
    y=1/(1+exp(-x));
end