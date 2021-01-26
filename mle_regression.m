clear
clc
close all

max=[57.8 62.2 68.4 76.0 84.3 88.2 90.7 90.7 86.9 79.5 68.0 60.5];
% max=[6.06 5.56 4.50 5.79 3.28 6.68 6.80 5.19 4.55 3.19 3.15 5.97];
N=length(max);
months=1:N;

fig=figure();
set(fig,'color','white')
plot(months,max,'b*')
grid on
hold on
xlabel('Month')
ylabel('Max Temp')

t=max';
x=months';
n=3;
X=[];
for idx=0:n
    X=[X,x.^idx];
end

w=(X'*X)\X'*t;
sigma=sqrt(1/N*(t'*t-t'*X*w));
y=normpdf(X*w,0,sigma);
plot(x',y','r-')