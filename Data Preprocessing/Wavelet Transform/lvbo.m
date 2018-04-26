% 滤波

data = xlsread('MyoRight-ztd-dajia-30-emg-1519732926.csv');
e = data(:,2);

%原始信号
subplot(2,2,1); 
plot(e) ;
xlabel('1 ：样本序列');
ylabel('原始信号幅值');
grid ;

%minimaxi原则小波去噪
s1=wden(e,'minimaxi','s','one',5,'db12');  
subplot(2,2,2);  
plot(s1); 
xlabel('2 ：db12降噪后信号' ) ;  
ylabel ('db12小波降噪后的信号幅值');  
grid; 

%heursure原则小波去噪
s2=wden(e,'heursure','s','one',5,'sym8');  
subplot(2,2,3);  
plot(s2);
xlabel('3 ：sym降噪后信号');  
ylabel('sym8小波降噪后的信号幅值'); 
grid;  

%sqtwolog原则小波去噪
s3=wden(e,'sqtwolog','s','one',5,'bior3.9');  
subplot(2,2,4);  
plot(s3);
xlabel('4 ：bior3.9降噪后信号');  
ylabel('bior3.9小波降噪后的信号幅值'); 
grid;  

