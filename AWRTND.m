function Z = AWRTND(M,Mask,option)

r=option.r;
maxiter=option.maxitr;
tol=option.stopc;
yita=option.yita;
sigmamin=option.sigmamin;

N=ndims(M);
Nway = size(M);
tempdim = r+r';  tempdim(tempdim==0) = []; 
tempdim = reshape(tempdim,N-1,N);   tempdim = tempdim'; 
tempdim(:,end+1) = Nway';


Z=cell(1,N);

for k=1:1:N
    Z{k}=0.001*rand(tempdim(k,:));
end

nZ_pre=0;

for itr=1:1:maxiter
    
    Ihat=tnprod_new(Z);
    err=Mask.*(M-Ihat);
    AAA=err(:);
    AAA(AAA==0)=[];
    Le=length(AAA);
    indxA=randperm(Le,min(20000,Le));
    AAA=AAA(indxA);
    kernelwidth=max(max(abs(quantile(AAA,0.25)),abs(quantile(AAA,0.75)))*yita,sigmamin);
    W=exp(-err.^2/2/kernelwidth^2);
    
    for k=1:1:N
        
        Zk=my_Unfold(Z{k},tempdim(k,:),N);
        Znk = tnreshape_new(tnprod_rest_new(Z,k),N);
        Y=Znk';    
        GY=Y'*Y;
        nGY=norm(GY(:));
        
        Mk=my_Unfold(M,Nway,k);
        Wk=my_Unfold(W,Nway,k);
        Maskk=my_Unfold(Mask,Nway,k);
        
        dY=-Wk.*Maskk.*(Mk-Zk*Y')*Y;
        dY2=-dY/(GY+1e-6*nGY*eye(size(GY,1)));
        dY3=sqrt(Wk.*Maskk).*(dY2*Y');
        tx=trace(dY'*dY2)/norm(dY3(:))^2;
        
        Zk=Zk-tx*dY2;
        Z{k}=my_Fold(Zk,tempdim(k,:),N);
        
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