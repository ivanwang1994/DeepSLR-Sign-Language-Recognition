clc;
clear all;
close all;
data = xlsread('MyoRight-ztd-dajia-30-emg-1519732926.csv');
b = data(:,2);
lenn=length(data(:,2));

nlen=input('请输入子向量长度nlen: '); %输入nlen
for indi=1:lenn-nlen+1 %起始点循环
    index=indi:indi+nlen-1;%
    average=sum(b(index))/nlen; %
    sig2(indi)=average; %赋值给信号2
end
% plot(data(:,1),a1);
% a = data(:,2)
%低通滤波
% b = smooth(aa,20,'moving');

%线性最小二乘滤波
% d = smooth(aa,30,'lowess');


% legend('噪声数据','默认中值滤波','20阶中值滤波')

% xx = medfilt1(a,20);

subplot(2,1,1); 
plot(b);
subplot(2,1,2); 
plot(sig2);
% plot(data(:,1),data(:,2))