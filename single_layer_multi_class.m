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

Weight=2*rand(size(correct_Output,2),input_row*input_col)-1;

for epoch=1:5000
    Weight=DeepLearning(Weight,input_Image,correct_Output);
end

% TESTING THE NEURAL NETWORK
test_input_Image=[1 1 1 0 1;1 1 0 0 1 ; 1 0 1 0 1;0 0 0 0 0;1 1 1 0 1;];
test_input_Image=reshape(test_input_Image,input_row*input_col,1);

input_of_layer=Weight*test_input_Image;
final_output=Softmax(input_of_layer)

% DEEP LEARNING RULE
function Weight=DeepLearning(Weight,input_Image,correct_Output)
    alpha=0.01; % learning rate
    for k=1:5
        reshaped_input_Image=input_Image(:,:,k);
        reshaped_input_Image=reshape(reshaped_input_Image,size(reshaped_input_Image,1)*size(reshaped_input_Image,2),1);
        
        input_of_layer=Weight*reshaped_input_Image;
        final_output=Softmax(input_of_layer);
        
        correct_Output_transpose=correct_Output(k,:)';
        error=correct_Output_transpose-final_output;
        
        delta=error;
        adjustment_of_weight=alpha*delta*reshaped_input_Image';
        Weight=Weight+adjustment_of_weight;
        
    end
end

function y=Softmax(x)
ex=exp(x);
y=ex/sum(ex);
end