function x=signals_test2(db_noise)

x=zeros(2000,15);

A=10;
phase_const=pi/4;
samples_delay=1;
N=1000;
start_point=100;



period_1=1:1100; period_2=1101:1100; period_3=1101:1200;
period_1_plus=1:(length(period_1)+samples_delay);

f1=10; f2=20; f3=15;


xs1(:,1)=A*cos(2*pi*(1:1200)*f1/1000); 

%A12=1; A13=0; A23=1; A24=0; A34=0; A14=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x(period_1,1)=xs1(100+period_1,1); 
xs=x(:,1);
%xs = 10*chirp((0.001:0.001:1.1),8,1,30)';
x(1:1100,1)=xs(1:1100,1);   x(:,1)=awgn(x(:,1),db_noise,'measured','db');

x(samples_delay+1:1100,2)=x(1:1100-samples_delay,1); x(:,2)=awgn(x(:,2),db_noise,'measured','db');
x(samples_delay+1:1100,3)=x(1:1100-samples_delay,1); x(:,3)=awgn(x(:,3),db_noise,'measured','db');
x(samples_delay+1:1100,4)=x(1:1100-samples_delay,2); x(:,4)=awgn(x(:,4),db_noise,'measured','db');

x(:,5)=0+0.5*wgn(2000,1,0);

x=x(start_point:end,:);
