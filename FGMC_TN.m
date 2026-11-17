function Out = FGMC_TN(G,rest)
N = length(G);
U = cell(1,N);
for n = 1:N
    R = size(G{n}); R(N) = [];
    temp = tensor_contraction(G{n},G{n},N,N);
    pairs = [1:N-1; N:2*(N-1)]; % 两行：[1,2,3,4; 5,6,7,8]
    order = reshape(pairs, 1, []); % 转置并拉直
    temp = permute(temp, order);
    U{n} = reshape(temp, R.^2);
end

m1 = zeros(1,N-2); m2 = zeros(1,N-2);
n = 1:N-1; 
if rest < N
   n(rest)=[];
end
for i=1:N-2
    m1(i)=1+(i-1)*(N-1);
    m2(i)=2+(i-1)*(N-1);
end
j=1;
if rest>1
    Out = U{1};
    for i=1:N-1
        if i+1 < rest
            Out = tensor_contraction(Out,U{i+1},m1(1:j),n(1:j));
            m1(2:end)=m1(2:end)-[1:N-3];
            m2(2:end)=m2(2:end)-[1:N-3];
            j=j+1;
        end
        if i+1 > rest
            Out = tensor_contraction(Out,U{i+1},m2(1:j),n(1:j));
            m2(2:end)=m2(2:end)-[1:N-3];
            j=j+1;
        end
    end
end
if rest == 1    
    Out = U{2};
    for i=2:N-1
        Out = tensor_contraction(Out,U{i+1},m2(1:i-1),n(1:i-1));
        m2(2:end)=m2(2:end)-[1:N-3];
    end
end
R = size(G{rest}); R(N) = [];
Rall = prod(R);
R2 = [R;R];
R2 = reshape(R2, 1, []); % 转置并拉直
Out = reshape(Out,R2);
order = [1:2:2*N-3,2:2:2*N-2];
Out = permute(Out,order);
Out = reshape(Out,[Rall,Rall]);
end




