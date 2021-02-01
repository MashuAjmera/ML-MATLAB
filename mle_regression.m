clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

load accidents % Importing the dataset
x = hwydata(:,14); % taking the 14th column as input data
t = hwydata(:,4); % taking the 4th scolumn as output data
N=length(x); % taking N as the number of instances

fig=figure(); % initializing fig for plot drawing
set(fig,'color','white')
plot(x,t,'b*') % plotting input vs output using blue stars
grid on % configuring to show grid
hold on
xlabel('Population') % x axis label
ylabel('Accident') % y axis label

n=1; % taking complexity of the model

% to compute the matrix for powers of train input column wise
X=[];
for idx=0:n
    X=[X,x.^idx];
end

% using the result from MLE for calculating w and sigma
w=(X'*X)\X'*t;
sigma=sqrt((t'*t-t'*X*w)/N);

z=X*w;
E=normpdf(z,0,sigma); % claculating the noise
y=z+E; % outputting the final value by adding noise to the answer
plot(x,y,'r-') % plotting the derived curve using red lines