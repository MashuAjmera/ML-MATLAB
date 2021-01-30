clear
clc
close all

load accidents
x = hwydata(:,14);
t = hwydata(:,4);
N=length(x);

fig=figure();
set(fig,'color','white')
plot(x,t,'b*')
grid on
hold on
xlabel('Population')
ylabel('Accident')

n=1;
X=[];
for idx=0:n
    X=[X,x.^idx];
end

w=(X'*X)\X'*t;
sigma=sqrt((t'*t-t'*X*w)/N);
z=X*w;
E=normpdf(z,0,sigma);
y=z+E;
plot(x,y,'r-')