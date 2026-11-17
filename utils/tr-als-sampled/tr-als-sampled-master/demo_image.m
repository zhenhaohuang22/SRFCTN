clear all

I=zeros(1080,1920,10);
for kk=11:30
    Temp=imread(['D:\Postdoc\MATRC\flamingo\000' num2str(kk) '.jpg']);
    I(:,:,kk-10)=double(rgb2gray(Temp))/255;
end

para_TR.max_tot     = 1e-3; para_TR.max_iter      =100;                     para_TR.disp =  1;
ObsRatio = 1;

P_Omega = zeros(size(I));
P_Omega( randsample( numel(I), round(ObsRatio * numel(I))) )=1; 
U_Omega = I.*P_Omega; 

idx=1:144;
MissM=U_Omega(idx,idx,:);
Mask=P_Omega(idx,idx,:);
nn=size(MissM);

n_t=[12,12,12,12,20];
MissM_t=reshape(MissM,n_t);
Mask_t=reshape(Mask,n_t);
N=ndims(MissM_t);

ranks = 30*ones(size(n_t));
target_acc = 1.2;
no_it = 30;

%% TR-ALS
% 
% no_it = 30;
% 
% tic; cores1 = tr_als(X, ranks, 'tol', 0, 'maxiters', no_it, 'verbose', true); time_1 = toc;
% Y = cores_2_tensor(cores1);
% acc_1 = norm(Y(:) - X(:))/norm(X(:));
% fprintf('TR-ALS accuracy is %.2e\n', acc_1)
% I_hat=reshape(Y,size(MissM));

%% TR-ALS-Sampled
X=MissM_t;
J = 2*ranks(1)^2;
J_inc = 300;
time_2 = [];
acc_2 = [];
while true
    tic; 
    cores2 = tr_als_sampled(X, ranks, J*ones(size(n_t)), 'tol', 0, 'maxiters', no_it, 'resample', true, 'verbose', true); 
    time_2 = [time_2; toc];
    Y = cores_2_tensor(cores2);
    acc_2 = [acc_2; norm(Y(:)-X(:))/norm(X(:))];
    fprintf('\tAccuracy for TR-ALS-Sampled is %.2e\n', acc_2(end));
    I_hat2=reshape(Y,size(MissM));
%     if acc_2(end)/acc_1 < target_acc
%         fprintf('\tTarget accuracy reached; breaking...\n');
%         break
%     end
    J = J + J_inc;
end

