function [X_f, Z_f, E_f, convergence] = DeFusion(D, lowDim, alpha, gamma, KNeigh, fout)
% @Input:
% dataCell: multiple data matrices stored a cell.
% lowDim: the number of dimensionality of latent sample representation.
% alpha and gamma: Parameters in DeFusion
% KNeigh: parameter for NE and SNF, usually set to be 20.
% fout: data path to save the output.

% @Output
% X: latent sample representation, a N x lowDim matrix. 
% Z: latent variables of features, a cell. 
% convergence: loss of objective function in each iteration, a structure array.
%% model
% min_{X, Z_{i}, E_{i}} \sum_{i=1}^{v}(\|D_{i} - XZ_{i} - E_{i}\|_{F}^{2} 
%        + \alpha\|E\|_{i}^{1}+\beta\|Z\|_{2,1}) + \lambda \|X\|_{1}+ \gamma trace(X^{T}LX) 
% s.t. X \ge 0, Z_{i} \ge 0
%%
%% parameter setting
inner = 10;
options.alpha = alpha;
options.beta = 1;
options.gamma = gamma;
options.lambda = 1;
options.nrep = 30;
options.step = 50;
options.maxiter = 600;
options.maxinner = inner;
options.silence = true;

beta = options.beta;
lambda = options.lambda;
maxiter = options.maxiter;
maxinner = options.maxinner;
nrep = options.nrep;
step = options.step;
silence = options.silence;

%% NE and SNF
[~, ~, ~, L] = snfCal(D, KNeigh);
%% record
convergenceObj = cell(nrep, 1);
convergenceRun = zeros(nrep, 1);
objVal = [];
%% repetition
for rp = 1:nrep
    initX = unifrnd(0, 2, [size(D{1}, 1), lowDim]);
    initZ = cell(length(D), 1);
    initE = cell(length(D), 1);
    for k = 1:length(initZ)
        % initZ{k} = zeros(lowDim, size(D{k}, 2));
        initZ{k} = unifrnd(0, 2, [lowDim, size(D{k}, 2)]);
        initE{k} = zeros(size(D{k}, 1), size(D{k}, 2));
    end
    X = initX;
    Z = initZ;
    E = initE;
    
    startEval = 0;
    oldEval = inf;
    newEval = 0;
    run = 1;
    objValOnRep = [];
    %% 
    while abs(oldEval - newEval) > (startEval - newEval)*1e-2 && (run <= maxiter)
        %% linearization of X
        for xinner =1:maxinner
            gradX = 2*gamma*L*X;
            for i=1:length(D)
                gradX = gradX - 2*(D{i} - X*Z{i} - E{i})*Z{i}';
            end
            ZZ = cellfun(@(C)C*C', Z, 'UniformOutput', 0);
            deltaX = 1/(norm(sum(cat(length(D), ZZ{:}),length(D)),'fro') + norm(L, 'fro'));
            X = proxL1Solver(X - deltaX*gradX, lambda*deltaX);
            X(X<0) = 0;
        end
        %% solver for Z
        for i=1:length(D)
            for zinner = 1:maxinner
                deltaZi = 1/norm(X'*X,'fro');
                gradZi = -2*X'*(D{i} - X*Z{i} - E{i});
                Zi = proxL21Solver(Z{i} - deltaZi*gradZi, beta*deltaZi);
                Zi(Zi<0) = 0;
                Z{i} = Zi;
            end
        end
        %% solver for E
        for i=1:length(D)
            E{i} =  proxL1Solver(D{i}-X*Z{i}, alpha/2);
        end        
        %% record obj
        if run == 1          
            obj = evalObj(D, X, Z, E, L, options);
            newEval = obj;
            startEval = newEval;
            if ~silence
                fprintf('run | obj |\n')
                fprintf('%d | %f |\n', run, newEval);
            end
        end
        if run ~=1 && mod(run, step) == 0
            oldEval = newEval;
            obj = evalObj(D, X, Z, E, L, options);
            newEval = obj;
            if ~silence
                fprintf('%d | %f |\n', run, newEval);
            end
        end
        objOnRp = evalObj(D, X, Z, E, L, options);
        objValOnRep(end+1) = objOnRp;
        run = run + 1;
    end
    %% record for this repetition
    objVal(end+1) = newEval;
    convergenceObj{rp} = objValOnRep;
    convergenceRun(rp) = run - 1;
    if ~silence
        fprintf('%d |%d| %f\n', rp, run-1, newEval);
    end
    if objVal(end) == min(objVal)
        X_f = X;
        Z_f = Z;
        E_f = E;
    end
end
convergence.Obj = convergenceObj;
convergence.Run = convergenceRun;
save(fout, 'convergence', 'X_f', 'Z_f', 'E_f');
end
%%
function [obj] = evalObj(D, X, Z, E, L, options)
    alpha = options.alpha;
    beta = options.beta;
    gamma = options.gamma;
    lambda = options.lambda;
    obj = gamma*trace(X'*L*X) + lambda*sum(sum(abs(X(:))));
    for i = 1:length(D)
        obj = obj + norm(D{i}-X*Z{i}-E{i},'fro')^2  ...
            + alpha*sum(sum(abs(E{i}))) + beta*L21Norm(Z{i});
    end
end
%%
function [L21] =  L21Norm(X, dim)

if nargin < 2
    dim = 1;
end

if dim == 1
    L21 = sum(arrayfun(@(n) norm(X(:, n)), 1:size(X,2)));
else
    L21 = sum(arrayfun(@(n) norm(X(n, :)), 1:size(X,1)));
end

end