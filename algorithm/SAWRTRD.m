function [Z,sampling_num_out,W]=SAWRTRD(M,Mask,option)

r=option.r;
sample_num=option.sample_num;
maxiter=option.maxitr;
tol=option.stopc;
yita=option.yita;
sigmamin=option.sigmamin;

N=ndims(M);
r(N+1)=r(1);



n_t=size(M);
W_new=ones(n_t);

Z=cell(1,N);

for k=1:1:N
    Z{k}=0.001*rand(r(k),size(M,k),r(k+1));
end

nZ_pre=0;

sampling_num_out=zeros(1,N);

for itr=1:1:maxiter

    W=W_new;
    
    for k=1:1:N

        Zk=reshape(permute(Z{k},[2 1 3]),size(Z{k},2),[]);

        Zs=cell(N,1);
        indX=cell(1,N);
        nk=[k+1:N 1:k-1];
        rZ=min((sample_num/prod(n_t(nk)))^(1/(N-1)),1);
        J=zeros(1,N);

        for p=1:1:N
            if p~=k
                indp=randperm(size(Z{p},2));
                sz=ceil(size(Z{p},2)*rZ);
                indX{p}=sort(indp(1:sz));
                Zs{p}=Z{p}(:,indX{p},:);
                J(p)=sz;
            else
                sz=size(Z{p},2);
                indX{p}=1:sz;
                J(p)=sz;
            end
        end
        
        Zsnk=TCP(Zs(nk));
        Ys=reshape(TensPermute(Zsnk,2),size(Zsnk,2),[]);
        sampling_num_out(k)=size(Ys,1);
              
        GY=FGMC(Z(nk));

        if N==3
            Ms=M(indX{1},indX{2},indX{3});
            Masks=Mask(indX{1},indX{2},indX{3});
        elseif N==4
            Ms=M(indX{1},indX{2},indX{3},indX{4});
            Masks=Mask(indX{1},indX{2},indX{3},indX{4});
        elseif N==5
            Ms=M(indX{1},indX{2},indX{3},indX{4},indX{5});
            Masks=Mask(indX{1},indX{2},indX{3},indX{4},indX{5});
        elseif N==6
            Ms=M(indX{1},indX{2},indX{3},indX{4},indX{5},indX{6});
            Masks=Mask(indX{1},indX{2},indX{3},indX{4},indX{5},indX{6});
        elseif N==7
            Ms=M(indX{1},indX{2},indX{3},indX{4},indX{5},indX{6},indX{7});
            Masks=Mask(indX{1},indX{2},indX{3},indX{4},indX{5},indX{6},indX{7});
        end
                
        Msk=tenmat_sb(Ms,k);
        Masksk=tenmat_sb(Masks,k);
        
        E=Masksk.*(Msk-Zk*Ys');
        
        err=reshape(E,J([k nk]));
        err=ipermute(err,[k nk]);
        AAA=err(:);
        AAA(AAA==0)=[];
        Le=length(AAA);
        indxA=randperm(Le,min(20000,Le));       
        AAA=AAA(indxA);
        kernelwidth=max(max(abs(quantile(AAA,0.25)),abs(quantile(AAA,0.75)))*yita,sigmamin);
        if N==3
            W_new(indX{1},indX{2},indX{3})=exp(-err.^2/2/kernelwidth^2);
        elseif N==4
            W_new(indX{1},indX{2},indX{3},indX{4})=exp(-err.^2/2/kernelwidth^2);
        elseif N==5
            W_new(indX{1},indX{2},indX{3},indX{4},indX{5})=exp(-err.^2/2/kernelwidth^2);
        end
        Wsk=exp(-E.^2/2/kernelwidth^2);

        dY=-Wsk.*E*Ys;
        nGY=norm(GY(:));
        dY2=-dY/(GY+1e-6*nGY*eye(size(GY,1)));
        dY3=sqrt(Wsk.*Masksk).*(dY2*Ys');  
        tx=trace(dY'*dY2)/norm(dY3(:))^2;

        Zk=Zk-tx*dY2;
        Z{k}=permute(reshape(Zk,[],r(k),r(k+1)),[2 1 3]);
                         
        
    end

    nZ=Zk*GY*Zk';
    err=norm(nZ(:)-nZ_pre(:))/norm(nZ(:));
    if err<tol&&itr>6
        fprintf('End at %d\n', itr);
        break;
    end
    nZ_pre=nZ;

end

end