function Z=AWRTRD_gradient(M,Mask,option)

r=option.r;
maxiter=option.maxitr;
tol=option.stopc;
yita=option.yita;
sigmamin=option.sigmamin;

N=ndims(M);
r(N+1)=r(1);

Z=cell(1,N);

for k=1:1:N
    Z{k}=0.001*rand(r(k),size(M,k),r(k+1));
end

nZ_pre=0;

for itr=1:1:maxiter
    
    TCP2=TCP(Z(2:end));
    Ihat=reshape(T2M_n(Z{1})*T2M_r(TCP2,2)',size(M));
    err=Mask.*(M-Ihat);
    AAA=err(:);
    AAA(AAA==0)=[];
    Le=length(AAA);
    indxA=randperm(Le,min(20000,Le));
    AAA=AAA(indxA);
    kernelwidth=max(max(abs(quantile(AAA,0.25)),abs(quantile(AAA,0.75)))*yita,sigmamin);
    W=exp(-err.^2/2/kernelwidth^2);  
    
    for k=1:1:N

        Zk=reshape(permute(Z{k},[2 1 3]),size(Z{k},2),[]);

        nk=[k+1:N 1:k-1];
        
        Znk=TCP(Z(nk));
        Y=reshape(TensPermute(Znk,2),size(Znk,2),[]);
                
        Mk=tenmat_sb(M,k);
        Wk=tenmat_sb(W,k);
        Maskk=tenmat_sb(Mask,k);

        dY=-Wk.*Maskk.*(Mk-Zk*Y')*Y;
        dY2=-dY;
        dY3=sqrt(Wk.*Maskk).*(dY2*Y');  
        tx=trace(dY'*dY2)/norm(dY3(:))^2;

        Zk=Zk-tx*dY2;
        Z{k}=permute(reshape(Zk,[],r(k),r(k+1)),[2 1 3]);
                
    end
       
    nZ=Zk*GY*Zk';
    err=norm(nZ(:)-nZ_pre(:))/norm(nZ(:));
    if err<tol
        fprintf('End at %d\n', itr);
        break;
    end
    nZ_pre=nZ;


end

end