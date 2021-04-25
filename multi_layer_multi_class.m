clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

T= readtable('.IRIS.csv');
[T.species,gN]=grp2idx(T.species);
n = size(T,1);
class=size(gN,1);

% Test train split parameters
idx = randperm(n);
ratio=0.9; % fraction of train set

% train set defined
input_Image=T{idx(1:round(ratio*n)),1:4};
y=T{idx(1:round(ratio*n)),5};
train=size(input_Image,1);
features=size(input_Image,2);

% test set defined
test_input_Image=T{idx(round(ratio*n)+1:end),1:4};
y_test=T{idx(round(ratio*n)+1:end),5};
test=size(test_input_Image,1); % number of test instances
predict=zeros(test,1);

% making a 2D array
correct_Output=zeros(train,class);
for i=1:train
    if y(i)==1
        correct_Output(i,:)=[1 0 0];
    elseif y(i)==2
        correct_Output(i,:)=[0 1 0];
    else
        correct_Output(i,:)=[0 0 1];
    end
end

w1=2*rand(20,features)-1; % input weights for first hidden layer with 20 neurons from input layer neurons
w2=2*rand(40,20)-1; % input weights for second hidden layer with 40 neurons
w3=2*rand(20,40)-1; % input weights for third hidden layer with 20 neurons
w4=2*rand(class,20)-1; % input weights for output layer with 1 neuron since binary classification

for epoch=1:10000
    [w1,w2,w3,w4]=DeepLearning(w1,w2,w3,w4,input_Image,correct_Output);
end

predict=zeros(test,1);
% TESTING THE NEURAL NETWORK
for i=1:test
    input_of_hidden_layer1=w1*test_input_Image(i,:)';
    output_of_hidden_layer1=ReLU(input_of_hidden_layer1);
    
    input_of_hidden_layer2=w2*output_of_hidden_layer1;
    output_of_hidden_layer2=ReLU(input_of_hidden_layer2);
    
    input_of_hidden_layer3=w3*output_of_hidden_layer2;
    output_of_hidden_layer3=ReLU(input_of_hidden_layer3);
    
    input_of_output_node=w4*output_of_hidden_layer3;
    [final_output,idx]=max(Softmax(input_of_output_node));
    predict(i)=idx;
end
fprintf('predicted values: ');
disp(predict');
fprintf('actual values: ');
disp(y_test');

% Calculating the error
count=0;
for i=1:test
    if predict(i)~=y_test(i)
        count=count+1;
    end
end
error=count/test;
fprintf('The error is = %8.3f \n',error);
fprintf('The accuracy is = %8.3f \n',1-error);

% DEEP LEARNING RULE
function [w1,w2,w3,w4]=DeepLearning(w1,w2,w3,w4,input_Image,correct_Output)
    alpha=0.01; % learning rate
    tt=size(input_Image,1);
    for k=1:tt
        reshaped_input_Image=input_Image(k,:)';
        
        % learning using forward propagation
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
        w1=w1+alpha*delta1*reshaped_input_Image';
        w2=w2+alpha*delta2*output_of_hidden_layer1';
        w3=w3+alpha*delta3*output_of_hidden_layer2';
        w4=w4+alpha*delta*output_of_hidden_layer3';
        
    end
end

function y=ReLU(x)
y=max(0,x);
end

function y=Softmax(x)
ex=exp(x);
y=ex/sum(ex);
end