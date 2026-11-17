function B=FGMC(U)

N=length(U);

%% compute Q1

Q=zeros(size(U{1},1)^2,size(U{1},3)^2);

L=size(U{1},3);

if mod(N,2)==0

    for k=1:1:L
        for j=1:1:L
            S1=squeeze(U{1}(:,:,k));
            S2=squeeze(U{1}(:,:,j));
            temp=S1*S2';
            Q(:,size(U{1},3)*(k-1)+j)=temp(:);
        end
    end

else
    
    for k=1:1:L
        for j=1:1:L
            S1=squeeze(U{1}(:,:,j));
            S2=squeeze(U{1}(:,:,k));
            temp=S1*S2';
            Q(:,size(U{1},3)*(k-1)+j)=temp(:);
        end
    end

end

%% compute Q2-QN

for n=2:1:N

    Qtemp=zeros(size(U{n},1)^2,size(U{n},3)^2);
    
    L=size(U{n},3);

    for k=1:1:L
        for j=1:1:L
            S1=squeeze(U{n}(:,:,k));
            S2=squeeze(U{n}(:,:,j));
            temp=S1*S2';
            Qtemp(:,size(U{n},3)*(k-1)+j)=temp(:);
        end
    end
    
    Q=Q*Qtemp;
      
end

B=zeros(size(U{1},1)*size(U{N},3));

for i=1:1:size(U{1},1)
    for j=1:1:size(U{N},3)
        temp=Q((i-1)*size(U{1},1)+1:i*size(U{1},1),(j-1)*size(U{N},3)+1:j*size(U{N},3))';
        B(size(U{N},3)*(i-1)+j,:)=temp(:)';
    end
end

end




