clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

% TRAINING
input_row=5;
input_col=5;
input_Image=zeros(input_row,input_col,5);
input_Image(:,:,1)=[1 0 0 1 1;1 1 0 1 1;1 1 0 1 1;1 1 0 1 1;1 0 0 0 1;];
input_Image(:,:,2)=[0 0 0 0 1;1 1 1 1 0;1 0 0 0 1;0 1 1 1 1 ;0 0 0 0 0;];
input_Image(:,:,3)=[0 0 0 0 1;1 1 1 1 0;1 0 0 0 1;1 1 1 1 0;0 0 0 0 0;];
input_Image(:,:,4)=[1 1 1 0 1;1 1 0 0 1 ; 1 0 1 0 1;0 0 0 0 0;1 1 1 0 1;];
input_Image(:,:,5)=[0 0 0 0 0;0 1 1 1 1;0 0 0 0 1;1 1 1 1 0;0 0 0 0 1;];

correct_Output=[1 0 0 0 0;0 1 0 0 0;0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1;];

w1=2*rand(20,input_row*input_col)-1;
w2=2*rand(20,20)-1;
w3=2*rand(20,20)-1;
w4=2*rand(size(correct_Output,2),20)-1;

for epoch=1:10000
    [w1,w2,w3,w4]=DeepLearning(w1,w2,w3,w4,input_Image,correct_Output);
end

% TESTING THE NEURAL NETWORK
test_input_Image=[1 0 0 1 1;1 1 0 1 1;1 1 0 1 1;1 1 0 1 1;1 0 0 0 1;];
test_input_Image=reshape(test_input_Image,input_row*input_col,1);

input_of_hidden_layer1=w1*test_input_Image;
output_of_hidden_layer1=ReLU(input_of_hidden_layer1);

input_of_hidden_layer2=w2*output_of_hidden_layer1;
output_of_hidden_layer2=ReLU(input_of_hidden_layer2);

input_of_hidden_layer3=w3*output_of_hidden_layer2;
output_of_hidden_layer3=ReLU(input_of_hidden_layer3);

input_of_output_node=w4*output_of_hidden_layer3;
final_output=Softmax(input_of_output_node)

% DEEP LEARNING RULE
function [w1,w2,w3,w4]=DeepLearning(w1,w2,w3,w4,input_Image,correct_Output)
    alpha=0.01; % learning rate
    for k=1:5
        reshaped_input_Image=input_Image(:,:,k);
        reshaped_input_Image=reshape(reshaped_input_Image,size(reshaped_input_Image,1)*size(reshaped_input_Image,2),1);
        
        input_of_hidden_layer1=w1*reshaped_input_Image;
        output_of_hidden_layer1=ReLU(input_of_hidden_layer1);
        
        input_of_hidden_layer2=w2*output_of_hidden_layer1;
        output_of_hidden_layer2=ReLU(input_of_hidden_layer2);
        
        input_of_hidden_layer3=w3*output_of_hidden_layer2;
        output_of_hidden_layer3=ReLU(input_of_hidden_layer3);
        
        input_of_output_node=w4*output_of_hidden_layer3;
        final_output=Softmax(input_of_output_node);
        
        correct_Output_transpose=correct_Output(k,:)';
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

function y=Softmax(x)
ex=exp(x);
y=ex/sum(ex);
end