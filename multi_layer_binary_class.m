clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

% training to get Weight
input=[0 0 1; 0 1 1; 1 0 1; 1 1 1;];
correct_Output=[0 0 1 1];

w1=2*rand(20,3)-1;
w2=2*rand(20,20)-1;
w3=2*rand(20,20)-1;
w4=2*rand(1,20)-1;

for epoch=1:10000
    [w1,w2,w3,w4]=SGD_method(w1,w2,w3,w4,input,correct_Output);
end

% testing using the Weight obtained from training
test_input=[0 0 1; 0 1 1; 1 0 1; 1 1 1;];
N=length(test_input);
for k=1:N
    test_input_Image=test_input(k,:)';
    
    input_of_hidden_layer1=w1*test_input_Image;
    output_of_hidden_layer1=ReLU(input_of_hidden_layer1);
    
    input_of_hidden_layer2=w2*output_of_hidden_layer1;
    output_of_hidden_layer2=ReLU(input_of_hidden_layer2);
    
    input_of_hidden_layer3=w3*output_of_hidden_layer2;
    output_of_hidden_layer3=ReLU(input_of_hidden_layer3);
    
    input_of_output_node=w4*output_of_hidden_layer3;
    final_output=Sigmoid(input_of_output_node)
end

% using Stochastic Gradient Descend as the Learning Rule
function [w1,w2,w3,w4]=SGD_method(w1,w2,w3,w4,input,correct_Output)
    alpha=0.9; % setting learning rate as 0.9
    N=length(input);
    for k=1:N        
        reshaped_input_Image=input(k,:)';
        
        input_of_hidden_layer1=w1*reshaped_input_Image;
        output_of_hidden_layer1=ReLU(input_of_hidden_layer1);
        
        input_of_hidden_layer2=w2*output_of_hidden_layer1;
        output_of_hidden_layer2=ReLU(input_of_hidden_layer2);
        
        input_of_hidden_layer3=w3*output_of_hidden_layer2;
        output_of_hidden_layer3=ReLU(input_of_hidden_layer3);
        
        input_of_output_node=w4*output_of_hidden_layer3;
        final_output=Sigmoid(input_of_output_node);
        
        correct_Output_transpose=correct_Output(k);
        error=correct_Output_transpose-final_output;
        
        delta=error;
        
        error_of_hidden_layer3=w4'*delta;
        delta3=(input_of_hidden_layer3>0).*error_of_hidden_layer3;
        
        error_of_hidden_layer2=w3'*delta3;
        delta2=(input_of_hidden_layer2>0).*error_of_hidden_layer2;
        
        error_of_hidden_layer1=w2'*delta2;
        delta1=(input_of_hidden_layer1>0).*error_of_hidden_layer1;
        
        adjustment_of_w4=alpha*delta*output_of_hidden_layer3';
        adjustment_of_w3=alpha*delta3*output_of_hidden_layer2';
        adjustment_of_w2=alpha*delta2*output_of_hidden_layer1';
        adjustment_of_w1=alpha*delta1*reshaped_input_Image';
        
        w1=w1+adjustment_of_w1;
        w2=w2+adjustment_of_w2;
        w3=w3+adjustment_of_w3;
        w4=w4+adjustment_of_w4; 
    end
end

function y=ReLU(x)
y=max(0,x);
end

% Sigmoid Function
function y=Sigmoid(x)
    y=1/(1+exp(-x));
end