clear all
clc
clf
%
% BP_SMALL 和LARGE是用一个算法，没用MATLAB的NN TOOLs
% 测试的时候先看SMALL，LARGE要跑半个小时
%


%采用三层BP网络结构
%输入层神经元数为5，隐含层神经元数为3，输出层神经元数为1

%最大迭代次数
maxcishu=100000;

%e为计算输出和样本实际输出差
%在内存中开辟maxcishu个存储空间
e=zeros(maxcishu,1);

% 输入数据维度5，输入节点数5
% maxp当日最高价序列
% minp当日最低价序列
% sp当日开盘价
% ep当日收盘价
% tnum当日成交量
% 调用数据
%shuju=xlsread('dm.xlsx', 'B1:K151');

shuju=importdata('BP_SMALL.xlsx');
sp=shuju.data(:,1)';
maxp=shuju.data(:,2)';
minp=shuju.data(:,3)';
tnum=shuju.data(:,10)';
ep=shuju.data(:,4)';

%
%shuju=importdata('300.xls');
%sp=shuju.data(:,0)';% sp当日开盘价
%maxp=shuju.data(:,1)';
%minp=shuju.data(:,2)';
%tnum=shuju.data(:,4)';
%ep=shuju.data(:,5)';% ep当日收盘价

%将数据集按照2:1分为训练样本集，和测试样本集
jishulength=length(ep);
jishu=ceil(jishulength/3*2) ;

%测试样本集是2/3处到最后一个
spt=sp(jishu+1:end);
maxpt=maxp(jishu+1:end);
minpt=minp(jishu+1:end);
tnumt=tnum(jishu+1:end);
ept=ep(jishu+1:end);

%训练样本集
sp=sp(1:jishu);
maxp=maxp(1:jishu);
minp=minp(1:jishu);
tnum=tnum(1:jishu);
ep=ep(1:jishu);

%记录下每组的最大值最小值，为训练样本集的归一化准备
maxp_max=max(maxp);
maxp_min=min(maxp);
minp_max=max(minp);
minp_min=min(minp);
ep_max=max(ep);
ep_min=min(ep);
sp_max=max(sp);
sp_min=min(sp);
tnum_max=max(tnum);
tnum_min=min(tnum);

% 目标数据为次日的收盘价，相当于把当日收盘价时间序列向前挪动一个单位
goalp=ep(2:jishu);

%数据归一化,将所有数据归一化到(0 1)
guiyi=@(A)((A-min(A))/(max(A)-min(A)));
maxp=guiyi(maxp);
minp=guiyi(minp);
sp=guiyi(sp);
ep=guiyi(ep);
tnum=guiyi(tnum);

% 后面的目标数据goalp个数是ep向前移动一位得到，所以最后一组的目标数据缺失
% 所以，要把除了目标数据goalp以外的所有数据序列删除最后一个
maxp=maxp(1:jishu-1);
minp=minp(1:jishu-1);
sp=sp(1:jishu-1);
ep=ep(1:jishu-1);
tnum=tnum(1:jishu-1);

%需要循环学习次数loopn，即训练样本的个数
loopn=length(maxp);
%为了方便表示将5个行向量放到一个5*loopn的矩阵中simp中,每一列是一个样本向量
simp=[maxp;minp;sp;ep;tnum];

%隐含层节点n
%根据相关资料，隐含层节点数比输入节点数少，一般取1/2输入节点数
bn=3;

%隐含层激活函数为S型函数
jihuo=@(x)(1/(1+exp(-x)));

%bx用来存放隐含层每个节点的输出
%bxe用来保存bx经过S函数处理的值，即输出层的输入
bx=zeros(bn,1);
bxe=zeros(bn,1);

%权值学习率u
u=0.02;

%W1(m,n)表示隐含层第m个神经元节点的第n个输入数值的权重，
%即，每一行对应一个节点
%所以输入层到隐含层的权值W1构成一个bn*5的矩阵，初值随机生成
W1=rand(bn,5);

%W2(m)表示输出节点第m个输入的初始权值，采用随机生成
W2=rand(1,bn);

%loopn个训练样本，对应loopn个输出
out=zeros(loopn,1);

for k=1:1:maxcishu
    
    %训练开始,i表示为本次输入的是第i个样本向量
    for i=1:1:loopn
        
        %求中层每个节点bx(n)的输出，系数对应的是W1的第n行
        for j=1:1:bn
            bx(j)=W1(j,:)*simp(:,i);
            bxe(j)=jihuo(bx(j));
        end
        
        %求输出
        out(i)=W2*bxe;
        
        %误差反向传播过程
        %计算输出节点的输入权值修正量,结果放在行向量AW2中
        %输出神经元激活函数 f(x)=x
        %为了书写方便，将deta用A代替
        AW2=zeros(1,bn);
        AW2=u*(out(i)-goalp(i))*bxe';
        
        %计算隐含层节点的输入权值修正量,结果放在行向量AW1中,需要对隐含层节点逐个处理
        AW1=zeros(bn,5);
        for j=1:1:bn
            AW1(j,:)=u* (out(i)-goalp(i))*W2(j)*bxe(j)*(1-bxe(j))*simp(:,i)';
        end
        W1=W1-AW1;
        W2=W2-AW2;
    end
    
    %计算样本偏差
    e(k)=sum((out-goalp').^2)/2/loopn;
    %误差设定
    if e(k)<=0.001
        disp('迭代次数')
        disp(k)
        disp('训练样本集误差')
        disp(e(k))
        break
    end
end

%显示训练好的权值
W1
W2
%绘制误差收敛曲线，直观展示收敛过程
figure(1)
hold on
e=e(1:k);
plot(e)
title('训练样本集误差曲线')
% 计算输出和实际输出对比图
figure(2)
plot(out,'rp')
hold on
plot(goalp,'bo')
title('训练样本集计算输出和实际输出对比')

%学习训练过程结束

%进行测试样本阶段,变量用末尾的t区分训练样本
maxpt=(maxpt-maxp_min)/(maxp_max-maxp_min);
minpt=(minpt-minp_min)/(minp_max-minp_min);
spt=(spt-sp_min)/(sp_max-sp_min);
eptduibi=ept(2:end);
ept=(ept-ep_min)/(ep_max-ep_min);
tnumt=(tnumt-tnum_min)/(tnum_max-tnum_min);

% 同样，将多维数据放入一个矩阵中，便于处理
simpt=[maxpt;minpt;spt;ept;tnumt];

%因为是用当前的数据预测下一天的，所以检验样本第一天的收盘价和预测的最后一天的收盘价因为没有比对值而舍弃
for i=1:1:length(maxpt)-1
    for j=1:1:bn
        bx(j)=W1(j,:)*simpt(:,i);
        bxe(j)=jihuo(bx(j));
    end
    
    %输出预测序列
    outt(i)=W2*bxe;
end

%预测输出和实际对比散点图
figure(3)
hold on
plot(outt,'rp')
plot(eptduibi,'bo')
title('测试样本集预测输出和实际对比')

%计算全局误差
disp('测试样本集误差')
disp(1/length(eptduibi)*0.5*sum((eptduibi-outt).^2))

