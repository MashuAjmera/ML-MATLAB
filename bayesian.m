clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

opts=detectImportOptions('results.csv');
opts.VariableTypes{8}='string';

t= readtable('results.csv',opts);
ind1 = t.Event == "10000M Men";
T=t(ind1,1:8);
T.Medal=grp2idx(T.Medal);
T.Nationality=grp2idx(T.Nationality);

for i=1:size(T,1)
    y=split(T.Result(i),":");
    T.Result(i)=str2double(string(y(1)))*60+str2double(string(y(2)));
end
T.Result=str2double(T.Result);
T=sortrows(T,'Year');

PriorMdl = bayeslm(3,'ModelType','conjugate','VarNames',["Year" "Medal" "Nationality"]);

fhs = 10; % Forecast horizon size
X = T{1:(end - fhs),PriorMdl.VarNames(2:end)};
y = T{1:(end - fhs),'Result'};
XF = T{(end - fhs + 1):end,PriorMdl.VarNames(2:end)}; % Future predictor data
yFT = T{(end - fhs + 1):end,'Result'};                % True future responses

PosteriorMdl = estimate(PriorMdl,X,y,'Display',false);

yF = forecast(PosteriorMdl,XF);

figure;
plot(T.Year,T.Result);
hold on
plot(T.Year((end - fhs + 1):end),yF)
h = gca;
hp = patch([T.Year(end - fhs + 1) T.Year(end) T.Year(end) T.Year(end - fhs + 1)],...
    h.YLim([1,1,2,2]),[0.8 0.8 0.8]);
uistack(hp,'bottom');
legend('Forecast Horizon','True Result','Forecasted Result','Location','NW')
title('Olympic 10000M Men Race Result Prediction');
ylabel('rGNP');
xlabel('Year');
hold off

frmse = sqrt(mean((yF - yFT).^2))