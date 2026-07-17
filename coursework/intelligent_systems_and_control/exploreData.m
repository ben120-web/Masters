%NB: You will need to modify this script to match the names if the variables in
%your dataset.
clear all
load Group12
whos
Date=Date01;
% finding the index for start and end of different years in the dataset.
s2014=find(year(Date)==2014,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');
X01train=X01(s2014:f2016,:);
X01test=X01(s2017:f2018,:);
Datetrain=Date(s2014:f2016,:);
Datetest=Date(s2017:f2018,:);
%%
figure(1); clf
plot(Datetrain,X01train(:,1))
xlabel('Time (days)');
ylabel('Load (MW)')
title('TIME 01 = 00:30-01:00');
%%
figure(2) ; clf % temperature variables 
%v=2 current temperature; v=3-11 temperaure xhrs earlier; v= 12:18 average 
temperature over the last x hours
vsel=[2  5 12];
plot(Datetrain,X01train(:,vsel))
xlabel('Time (days)');
ylabel('selected variables')
legend(labels(vsel))
title('01 = 00:30-01:00');
%%
figure(3) ; clf  %v=19-22 wind related variables
for v=19:1:22;
plot(Datetrain,X01train(:,v))
xlabel('Time (days)');
ylabel(labels(v))
title('TIME 01 = 00:30-01:00');
pause(2)
end
%%
figure(4) ; clf  %v=19-22 wind related variables
for v=23:1:36;
plot(Datetrain,X01train(:,v))
xlabel('Time (days)');
ylabel(labels(v))
title('TIME 01 = 00:30-01:00');
pause(4)
end
%% looking at the correlation between variables
figure(5); clf
for v=2:36;
plot(X01train(:,v),X01train(:,1),'.')
xlabel(labels(v))
ylabel(labels(1))
r=corr(X01train(:,v),X01train(:,1));  %determines the correlation coefficeint 
between the variables.
title(sprintf('r = %1.3f',r));
pause
end
