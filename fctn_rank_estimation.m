function [R, info] = fctn_rank_estimation(X, delta, opts)
%FCTN_RANK_ESTIMATION  Algorithm 1: FCTN rank estimation (论文式(8))
%
% 输入
%   X     : N 阶张量 (数值/logical, 内部转为 double); 矩阵视为 N = 2
%   delta : 预设相对误差, 需满足 0 < delta < 1 (全流程统一使用)
%   opts  : (可选) 结构体
%           .mode    : 'svd'  (默认, 精确 SVD)
%                      'svds' (只求前 k 个奇异值, 适合大展开矩阵)
%                      'rand' (随机化投影 SVD, 适合超大展开矩阵)
%           .k       : 'svds'/'rand' 模式的探测秩数, 默认 200
%           .p       : 'rand' 模式的过采样数, 默认 10
%           .verbose : 是否打印每步结果, 默认 false
%
% 输出
%   R    : N x N 上三角矩阵, R(n1,n2) = R_{n1,n2}  (1 <= n1 < n2 <= N)
%          其余元素为 0, 与论文式(2)的写法一致
%   info : 诊断信息
%          .singleRank : 1 x N, 单模展开秩 r1(n)  = rank(X^{<n>})
%          .doubleRank : N x N, 双模展开秩 r12(n1,n2) = rank(X^{<n1,n2>})
%          .delta / .mode
%          .lemma1Err  : 1 x N, Lemma 1 相对残差
%          .lemma2Err  : N x N, Lemma 2 相对残差 (上三角有效)
%          .rankVector : 按 R12,R13,...,R1N,R23,... 顺序展平的秩向量
%          .time       : 各阶段耗时(秒)
%

%% ---------- 0. 输入解析与检查 ----------
if nargin < 2 || isempty(delta)
    delta = 1e-1;
end
if nargin < 3
    opts = struct();
end
opts = local_defaults(opts);

if ~(isnumeric(X) || islogical(X))
    error('fctn_rank_estimation:badInput', 'X 必须是数值张量。');
end
X = double(X);
if ~(isscalar(delta) && isreal(delta) && delta > 0 && delta < 1)
    error('fctn_rank_estimation:badDelta', 'delta 必须满足 0 < delta < 1。');
end
if any(~isfinite(X(:)))
    warning('fctn_rank_estimation:nonfinite', 'X 含 Inf/NaN, 秩估计结果不可靠。');
end

N  = ndims(X);
sz = size(X);

if N < 2                                        % 退化情形: 向量
    R    = zeros(N);
    info = struct('singleRank', local_truncated_rank(X(:), delta, opts), ...
                  'doubleRank', zeros(N), 'delta', delta, 'mode', opts.mode, ...
                  'lemma1Err', 0, 'lemma2Err', zeros(N), ...
                  'rankVector', zeros(1, 0), ...
                  'time', struct('single', 0, 'pairs', 0, 'total', 0));
    return;
end

tAll = tic;
meanX = mean(X(:));
X(X==0) = meanX;
%% ---------- 1. 单模展开矩阵的秩  (Algorithm 1 第 1-2 行) ----------
t0 = tic;
singleRank = zeros(1, N);
for n1 = 1:N
    M = local_mode_unfold(X, sz, n1);                     % X^{<n1>}
    singleRank(n1) = local_truncated_rank(M, delta, opts);
    if opts.verbose
        fprintf('[Algorithm 1] single-mode  n1 = %d : rank = %d  (size %d x %d)\n', ...
                n1, singleRank(n1), size(M, 1), size(M, 2));
    end
end
tSingle = toc(t0);

%% ---------- 2. 双模展开矩阵的秩 + 式(8)  (第 3-7 行) ----------
t0 = tic;
R          = zeros(N, N);
doubleRank = zeros(N, N);
for n1 = 1:N
    for n2 = n1 + 1:N
        M = local_mode_unfold(X, sz, [n1 n2]);            % X^{<n1,n2>}
        doubleRank(n1, n2) = local_truncated_rank(M, delta, opts);

        r1 = singleRank(n1);
        r2 = singleRank(n2);
        rD = doubleRank(n1, n2);

        if rD <= 0                                         % 全零展开矩阵保护
            R(n1, n2) = 1;
        else
            R(n1, n2) = round( sqrt( r1 * r2 / rD ) );      % 式(8), 取整即四舍五入
        end
        R(n1, n2) = max(2, min(R(n1, n2), min(size(M))));   % 合法范围约束

        if opts.verbose
            fprintf(['[Algorithm 1] double-mode  (n1,n2) = (%d,%d) : ' ...
                     'rank = %d  ->  R(%d,%d) = %d\n'], ...
                    n1, n2, rD, n1, n2, R(n1, n2));
        end
    end
end
tPair  = toc(t0);
tTotal = toc(tAll);

%% ---------- 3. 一致性检验 (Lemma 1 / Lemma 2) ----------
% Lemma 1: rank(X^{<n>}) = prod_{k<n} R_{k,n} * prod_{k>n} R_{n,k}
lemma1Err = nan(1, N);
for n = 1:N
    prodR = 1;
    for k = 1:n-1, prodR = prodR * R(k, n); end
    for k = n+1:N, prodR = prodR * R(n, k); end
    if singleRank(n) > 0
        lemma1Err(n) = abs(prodR - singleRank(n)) / singleRank(n);
    end
end

% Lemma 2: rank(X^{<n1,n2>}) = r1(n1) * r1(n2) / R_{n1,n2}^2
lemma2Err = nan(N, N);
for n1 = 1:N
    for n2 = n1 + 1:N
        predR = singleRank(n1) * singleRank(n2) / max(R(n1, n2)^2, eps);
        if doubleRank(n1, n2) > 0
            lemma2Err(n1, n2) = abs(predR - doubleRank(n1, n2)) / doubleRank(n1, n2);
        end
    end
end

%% ---------- 4. 打包输出 ----------
info               = struct();
info.singleRank    = singleRank;
info.doubleRank    = doubleRank;
info.delta         = delta;
info.mode          = opts.mode;
info.lemma1Err     = lemma1Err;
info.lemma2Err     = lemma2Err;
info.rankVector    = local_upper_tri_vector(R);
info.time          = struct('single', tSingle, 'pairs', tPair, 'total', tTotal);
end

%% ===================== 局部函数 =====================
function opts = local_defaults(opts)
% 补齐默认选项
def = struct('mode', 'svds', 'k', 100, 'p', 10, 'verbose', false);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i}) || isempty(opts.(f{i}))
        opts.(f{i}) = def.(f{i});
    end
end
if ~ismember(opts.mode, {'svd', 'svds', 'rand'})
    error('opts.mode 只能是 ''svd''、''svds'' 或 ''rand''。');
end
end

function M = local_mode_unfold(X, sz, modes)
% 计算展开矩阵 X^{<modes>}: 把 modes 指定的模换到最前, 再展平为矩阵
% 行指标顺序与论文一致: i_{n1} 变化最快, i_{n2} 次之
N     = numel(sz);
order = [modes, setdiff(1:N, modes)];
M     = reshape(permute(X, order), prod(sz(modes)), []);
end

function r = local_truncated_rank(M, delta, opts)
% delta-截断 SVD 秩估计: 最小的 r 使尾部能量 <= delta^2 * 总能量
if isempty(M)
    r = 1; return;
end
m = min(size(M));
if m < 1
    r = 1; return;
end

switch opts.mode
    case 'svd'                                    % 精确 SVD
        s   = svd(M, 'econ');
        tot = sum(s.^2);
        r   = local_min_rank(double(s(:)), delta, tot, m);

    case 'svds'                                   % 只求前 k 个奇异值
        k   = max(1, min(opts.k, m));
        s   = sort(double(svds(M, k, 'largest')), 'descend');
        tot = norm(M, 'fro')^2;                   % 廉价的总能量代理
        r   = local_min_rank(s, delta, tot, m);

    case 'rand'                                   % 随机化投影 SVD
        k   = max(1, min(opts.k, m));
        l   = max(1, min(k + opts.p, min(size(M))));
        Om  = randn(size(M, 2), l);
        [Q, ~] = qr(M * Om, 0);
        s   = sort(double(svd(Q' * M, 'econ')), 'descend');
        tot = norm(M, 'fro')^2;
        r   = local_min_rank(s, delta, tot, m);
end
end

function r = local_min_rank(s, delta, tot, upper)
% 最小 r 使 sum_{j>r} s_j^2 <= delta^2 * tot   (s 需为降序奇异值)
s = s(:);
if ~(tot > 0) || isempty(s)
    r = 1; return;
end
c   = cumsum(s.^2);
idx = find(c >= (1 - delta^2) * tot, 1, 'first');   % 累积能量首次达标的位置
if isempty(idx)
    idx = numel(s);
end
r = min(max(idx, 1), upper);
end

function v = local_upper_tri_vector(R)
% 按论文式(2)的书写顺序展平: R12,R13,...,R1N,R23,...,R(N-1,N)
N = size(R, 1);
v = zeros(1, N * (N - 1) / 2);
c = 0;
for n1 = 1:N-1
    for n2 = n1 + 1:N
        c = c + 1;
        v(c) = R(n1, n2);
    end
end
end
