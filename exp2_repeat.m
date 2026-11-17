clear all
addpath(genpath(pwd))

rv = [80,80,3,20];
c  = 0.2;
p  = 0.3;
var = 0.5;

imgDir = './yuz';
D = dir(fullfile(imgDir,'*.mat'));

% ===== options (do NOT set option.I here; X not defined yet) =====
option.r = rv;
option.maxitr = 30;
option.stopc = 1e-3;
option.sample_num = 30000;
option.embedding_dims = [200,200,3,10];
option.yita = 4;
option.sigmamin = 0.15;
option.estimator = 'cauchy';

numRepeat = 5;
numAlgs = 2; % 1:SAWRTND 2:SAWRTRD 3:RFCTN 4:FCTN 5:TW 6:BRTR 7:NTRC 8:RTRC

for nn = 1:numel(D)

    fprintf('Processing %d / %d\n', nn, numel(D))

    imgName = fullfile(imgDir, D(nn).name);
    S = load(imgName);

    % ---- data ----
    I = S.X;
    I = I(:,:,:,1:50);
    X = double(I) / 255;
    N = ndims(X);

    % ---- predefine per-repeat stats ----
    PSNR_all_repeat = zeros(numRepeat, numAlgs);
    time_all_repeat = zeros(numRepeat, numAlgs);

    for rep = 1:numRepeat
        % different randomness for each (nn, rep)
        rng(1000*rep + rep, 'twister');
        fprintf('  Repeat %d / %d\n', rep, numRepeat)

        % ===== generate noise =====
        if nn == 1
            G = noisemix(numel(X), 1, c, 0.001, var, 'gaussian');
            X_n = X + reshape(G, size(X));
            X_n = min(max(X_n, 0), 1);
        elseif nn == 2
            X_n = imnoise(X, 'salt & pepper', c);
        else
            idxn = randperm(numel(X), floor(c*numel(X)));
            X_n = X;
            X_n(idxn) = rand(length(idxn), 1);
        end

        % ===== missing mask =====
        Mask = ones(size(X));
        idxm = randperm(numel(X), floor((1-p)*numel(X)));
        Mask(idxm) = 0;

        MissM = X_n .* Mask;

        % ===== common opts that depend on X =====
        option.I = X;     % if your methods need it
        opt_tn = option;
        opt_tn.r = [0,  9,  2,  9;
                    0,  0,  2,  9;
                    0,  0,  0,  2;
                    0,  0,  0,  0];

        % ========= 1) SAWRTND =========
        t1 = tic;
        [Z, ~, ~] = SAWRTND(MissM, Mask, opt_tn);
        I_SAWRTND = reshape(tnprod_new(Z), size(X));
        PSNR_all_repeat(rep,1) = psnr(I_SAWRTND, X);
        time_all_repeat(rep,1) = toc(t1);

        % ========= 2) SAWRTRD =========
        t1 = tic;
        [Z, ~, ~] = SAWRTRD(MissM, Mask, option);
        TCP2 = TCP(Z(2:end));
        I_SAWRTRD = reshape(T2M_n(Z{1}) * T2M_r(TCP2,2)', size(X));
        PSNR_all_repeat(rep,2) = psnr(I_SAWRTRD, X);
        time_all_repeat(rep,2) = toc(t1);

    end % rep

    % ===== aggregate =====
    PSNR_mean = mean(PSNR_all_repeat, 1);
    PSNR_std  = std(PSNR_all_repeat, 0, 1);
    time_mean = mean(time_all_repeat, 1);
    time_std  = std(time_all_repeat, 0, 1);


end
