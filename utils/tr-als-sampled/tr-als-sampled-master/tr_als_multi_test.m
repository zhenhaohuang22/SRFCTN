function [cores, varargout] = tr_als_multi_test(X, ranks, I, varargin)
%tr_als Compute tensor ring decomposition via alternating least squares
%
%cores = tr_als(X, ranks) computes a tensor ring (TR) decomposition of the
%input N-dimensional array X. ranks is a length-N vector containing the
%target ranks. The output cores is a cell containing the N cores tensors,
%each represented as a 3-way array.
%
%cores = tr_als(___, 'tol', tol) is an optional argument that controls the
%termination tolerance. If the change in the relative error is less than
%tol at the conclusion of a main loop iteration, the algorithm terminates.
%Default is 1e-3. 
%
%cores = tr_als(___, 'maxiters', maxiters) is an optional argument that
%controls the maximum number of main loop iterations. The default is 50.
%
%cores = tr_als(___, 'verbose', verbose) is an optional argument that
%controls the amount of information printed to the terminal during
%execution. Setting verbose to true will result in more print out. Default
%is false.
%
%This algorithm is based on Alg. 2 in paper by
%Q. Zhao, G. Zhou, S. Xie, L. Zhang, A. Cichocki. "Tensor ring
%decomposition". arXiv:1606.05535, 2016.

%% Handle optional inputs

params = inputParser;
addParameter(params, 'conv_crit', 'none');
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', false, @isscalar);
parse(params, varargin{:});

conv_crit = params.Results.conv_crit;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;

%% Initialize cores
N=ndims(X);
r=ranks;
r(N+1)=r(1);
sz = size(X);
cores = initialize_cores(sz, ranks);

if nargout > 1 && tol > 0 && (strcmp(conv_crit, 'relative error') || strcmp(conv_crit, 'norm'))
    conv_vec = zeros(1, maxiters);
end

%% Main loop
% Iterate until convergence, for a maximum of maxiters iterations

N = length(sz);
er_old = Inf;
for it = 1:maxiters
    for n = 1:N
        
         Zs=cell(N,1);
         indX=cell(1,N);
        
         for p=1:1:N
            if p~=n
                indp=randperm(size(cores{p},2));
                indX{p}=sort(indp(1:floor(size(cores{p},2)*0.4)));
                Zs{p}=cores{p}(:,indX{p},:);
            else
                indX{p}=1:size(cores{p},2);
                Zs{p}=cores{p}(:,indX{p},:);
            end
        end
        
        % Compute G_{[2]}^{\neq n} from cores
        G = subchain_matrix(Zs, n);
        
        Xs=X(indX{1},indX{2},indX{3});
        % Compute relevant unfolding of data tensor X
        XnT = mode_unfolding(Xs, n).';
        
        % Solve LS problem and update core
        Z = XnT'*G/(G'*G);%(G \ XnT).';
        cores{n} = permute(reshape(Z,[],r(n),r(n+1)),[2 1 3]);%classical_mode_folding(Z, 2, size(cores{n}));        
    end
    
    TCP2=TCP(cores(2:end));
    Ihat=reshape(T2M_n(cores{1})*T2M_r(TCP2,2)',size(I));
    disp(psnr(Ihat,I));   
   
end

if nargout > 1 && exist('conv_vec', 'var')
    varargout{1} = conv_vec(1:it);
else
    varargout{1} = nan;
end

end
