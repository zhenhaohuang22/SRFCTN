function Mask = generate_mask(sz,p,type)

switch lower(type)

    %% ================= Existing Masks =================
    case 'block'
        S = load('block.mat');
        Mask = S.mask;

    case 'letter'
        S = load('letter.mat');
        Mask = S.mask;

    case 'obiliquel'
        S = load('obiliqueL.mat');
        Mask = S.mask;

    case 'obiliques'
        S = load('obiliqueS.mat');
        Mask = S.mask;

    case 'scratch'
        S = load('scratch.mat');
        Mask = S.mask;

    case 'stripe'
        S = load('stripe.mat');
        Mask = S.mask;

    %% ================= Random =================
    case 'random'

        Mask = ones(sz);

        idx = randperm(numel(Mask),...
            floor((1-p)*numel(Mask)));

        Mask(idx) = 0;

    otherwise
        error('Unknown mask type.');
end

end