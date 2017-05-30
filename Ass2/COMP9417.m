%function [PD,ED] = yuce_day(units,accuracy)
units = 10;
accuracy=30;

A=importdata('A.mat');%载入供预测数据
p1=A(1:43999,1:4); %input 
%A里面一共6列， 开盘／最高／最低／收盘／成交量／持仓数
%如果把成交量和持仓 一起输入波动率会变得很大
t1=A(2:44000,1); %target 训练集取四分之三
p=p1';
t=t1';
[pn,minp,maxp,tn,mint,maxt]=premnmx(p,t);%对要用以学习的数据作归一化处理
%[Pn,minp,maxp,Tn,mint,maxt]=premnmx(P,T)，其中P，T分别为原始输入和输出数据。
% PN - R x Q 矩阵 （归一化的输入向量）. 
% minp- R x 1 向量，包含对于P的最小值. 
% maxp- R x 1 向量，包含P的最大值. 
% TN - S x Q 矩阵，归一化的目标向量. 
% mint- S x 1 向量，包含每个目标值T的最小值。
% maxt- S x 1 向量，包含每个目标值T的最大值

%建立BP网络，并初始化训练参数
net=newff(minmax(pn),[accuracy,1],{'tansig','purelin'},'traingdm');

inputWeights=net.IW{1,1};
inputbias=net.b{1};
layerWeights=net.IW{1,1};
layerbias=net.b{2};
%变量初始化
%InData=6;             %输入层
%NeroData=10;          %隐层神经元个数
%OutData=1;            %输出数
LearnSpeed=0.01;      %学习速度
Display=50;           %显示次数
MaxTrain=50000;         %最大训练次数
Error=0.001;           %均方误差
Time=300;             %最多耗时(s)
ILR=10;               %学习速度增加率
DLR=0.1;              %学习速度减少率
MC=0.01;               %动量

net.trainparam.show=Display;
net.trainparam.epochs=MaxTrain;
net.trainparam.lr=LearnSpeed;
net.trainparam.goal=Error;
net.trainParam.time=Time;
net.trainParam.lr_inc=ILR;
net.trainParam.lr_dec=DLR;
net.trainParam.mc=MC;

%利用traingdm训练函数对网络进行自学习训练
net=train(net,pn,tn); %训练之后的net



%开始测试
%预测出44001-54001
p2=A(44000:54000,1:4); %测试
p2=p2';
p2n=tramnmx(p2,minp,maxp);%对p2归一化处理
a2n=sim(net,p2n);%仿真预测，
a2=postmnmx(a2n,mint,maxt);%对a2n进行反归一化处理
%ED=abs(a2'-A(16:2000,1));%记录误差数据

t_plot=A(44001:54001,1); %target 
t_plot=t_plot';



figure(1)

plot(t_plot,'b');
hold on;
plot(a2,'r');
title('SET OF TEST');


figure(2)
title('SET OF TRAINING');
Train_out=sim(net,pn);%仿真预测，
Train_out=postmnmx(Train_out,mint,maxt);%对a2n进行反归一化处理
plot(t,'b');
hold on;
plot(Train_out,'r');
title('SET OF TEST');


ED= t_plot-a2;
figure(3)
plot(ED)

;
%a2     1*10001 40000:54000 输出下一个min的预测
%t_plot 1*10001 40001-54000
t_ori= A(44000:54000,1);
t_ori= t_ori'
t_diff=t_plot-t_ori;
aNt_diff=a2-t_ori;
%a2o 预测出44000-54000
p2o=A(43999:53999,1:4); %测试
p2o=p2o';
p2no=tramnmx(p2o,minp,maxp);%对p2归一化处理
a2no=sim(net,p2no);%仿真预测，
a2o=postmnmx(a2no,mint,maxt);%对a2n进行反归一化处理
aNa_diff=a2-a2o;
aNa_d=heaviside(aNa_diff);
%0-1
t_d=heaviside(t_diff);
aNt_d=heaviside(aNt_diff);
%final_daNt=xor(t_d,aNt_d); %0为预测方向正确，1为反方向
final_d=xor(t_d,aNa_d); %0为预测方向正确，1为反方向

figure(4)
plot(final_d);

tabulate(final_d);
% 最后会输出，因为我们只要做多，或者做空，所以只要判断方向，不需要知道具体的大小
%  Value    Count   Percent
%      0     7784     77.83%
%      1     2217     22.17%

