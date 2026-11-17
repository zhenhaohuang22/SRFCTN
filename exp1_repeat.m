 clear all
addpath(genpath(pwd))

rv=[5,5,3];
c=0.2;
var=0.5;
p=0.3;

imgDir = './BSD500';
D= dir(fullfile(imgDir,'*.jpg'));

option.r=rv;
option.maxitr=30;
option.stopc=1e-3;
option.sample_num=30000;
option.embedding_dims=[400,400,3];
option.yita = 4; %huber2
option.sigmamin = 0.15; %huber0.05
option.estimator = 'cauchy';

numRepeat = 5;  % 重复次数

for nn=1:numel(D)
    disp(['Processing image ' num2str(nn) ' / ' num2str(numel(D))])
    imgName = fullfile(imgDir,D(nn).name);
    I=imread(imgName);
    I = I(1:320,1:480,:);
    X=double(I)/255;
    n_t=size(X);
    N=ndims(X);

    % 用于累积结果
    numAlgs = 2;  % SAWRTND, SAWRTRD, RFCTN, FCTN, TW, BRTR, NTRC, RTRC, FCTNFR（如果有）
    PSNR_all_repeat = zeros(numRepeat,numAlgs);
    time_all_repeat = zeros(numRepeat,numAlgs);

    for rep=1:numRepeat
        rng(1000*rep + rep, 'twister');
        disp(['  Repeat ' num2str(rep) ' / ' num2str(numRepeat)])
        % ====== 生成噪声 ======
        if nn<6
            G=noisemix(numel(X),1,c,0.001,var,'gaussian');
            X_n=X+reshape(G,size(X));
            X_n(X_n>1)=1; X_n(X_n<0)=0;
        elseif nn>=6 && nn<11
            X_n=imnoise(X,'salt & pepper', c);
        else
            indx=randperm(numel(X),floor(c*numel(X)));
            X_n=X; X_n(indx)=rand(floor(c*numel(X)),1);
        end

        % ====== 构造缺失mask ======
        Mask=ones(size(X));
        indx=randperm(numel(X),floor((1-p)*numel(X)));
        Mask(indx)=0;

        MissM = X_n.*Mask;
        MissM5 = reshape(MissM,[16,20,20,24,3]);
        Mask5 = reshape(Mask,[16,20,20,24,3]);
       
        %% 1. SAWRTND
        t1=tic;
        opt_tn = option;
        opt_tn.r = [0,4,4,4,2;0,0,4,4,2;0,0,0,4,2;0,0,0,0,2;0,0,0,0,0];
        [Z,opt_tn.sample_num_multi,~]=SAWRTND(MissM5,Mask5,opt_tn);
        I_SAWRTND = reshape(tnprod_new(Z),size(X));
        PSNR_all_repeat(rep,1) = psnr(I_SAWRTND,X);
        time_all_repeat(rep,1) = toc(t1);
        
        %% 2. SAWRTRD
        t1=tic;
        [Z,option.sample_num_multi,~]=SAWRTRD(MissM,Mask,option);
        TCP2=TCP(Z(2:end));
        I_SAWRTRD=reshape(T2M_n(Z{1})*T2M_r(TCP2,2)',size(X));
        PSNR_all_repeat(rep,2) = psnr(I_SAWRTRD,X);
        time_all_repeat(rep,2) = toc(t1);

    end

    % 计算平均值和标准差
    PSNR_mean = mean(PSNR_all_repeat,1);
    PSNR_std  = std(PSNR_all_repeat,0,1);
    time_mean = mean(time_all_repeat,1);
    time_std  = std(time_all_repeat,0,1);


end
