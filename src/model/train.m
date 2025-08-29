function [net, hist] = train(net, sys, tspan, ic, cfg, data)
    %% ---- Documentation ----

    % TRAIN  PINN training loop (mini-batch, momentum, gradient clipping).
    %        optional Adam optimizer and LR scheduler.
    %
    % [net, hist] = TRAIN(net, sys, [t0 t1], ic, cfg, data)
    %
    % cfg fields (defaults shown if missing):
    %   epochs        = 1500
    %   batch_size    = 128
    %   collocation_N = 4096
    %   seed          = 42
    %   lr            = 1e-3
    %   momentum      = 0.9                 (used for SGD fallback)
    %   grad_clip     = 5.0                 (0 disables)
    %   loss_weights  = struct('lambda_res',1,'lambda_ic',1,'lambda_data',0)
    %   optimizer     = 'adam' | 'sgd'      ('adam' enables Adam)
    %   beta1         = 0.9                 (Adam)
    %   beta2         = 0.999               (Adam)
    %   eps           = 1e-8                (Adam)
    %   lr_decay      = [] or 0.98          (optional scheduler)
    %   decay_every   = [] or 200           (epochs; scheduler step)
    %
    % Returns history with fields: epoch, loss, res, ic, data)

    %% ---- Train ----

    % defaults (robust to missing fields)
    if nargin < 6, data = []; end
    if nargin < 5 || isempty(cfg), cfg = struct(); end
    
    cfg = fill_defaults(cfg, struct( ...
        'epochs',        2000, ...
        'batch_size',    128, ...
        'collocation_N', 4096, ...
        'seed',          42, ...
        'lr',            1e-3, ...
        'momentum',      0.9, ...
        'grad_clip',     5.0, ...
        'loss_weights',  struct('lambda_res',1,'lambda_ic',1,'lambda_data',0), ...
        'optimizer',     'sgd', ...     % 'adam' to use Adam
        'beta1',         0.9, ...
        'beta2',         0.999, ...
        'eps',           1e-8, ...
        'lr_decay',      [], ...        % e.g. 0.98
        'decay_every',   [] ...         % e.g. 200
    ));
    
    % pre-sample collocation pool
    rng(cfg.seed);
    t_pool = collocation(tspan, cfg.collocation_N, 'random'); % 1 x Nc (dlarray)
    Nc = size(t_pool,2);
    
    % momentum / Adam buffers
    vel = struct();           % for SGD
    m   = struct(); v = struct();  % for Adam
    
    fn = fieldnames(net.params);
    for k = 1:numel(fn)
        vel.(fn{k}) = zeros(size(net.params.(fn{k})), 'like', net.params.(fn{k}));
        m.(fn{k})   = zeros(size(net.params.(fn{k})), 'like', net.params.(fn{k}));
        v.(fn{k})   = zeros(size(net.params.(fn{k})), 'like', net.params.(fn{k}));
    end
    
    num_batches = ceil(Nc / cfg.batch_size);
    
    % history
    hist.epoch = zeros(cfg.epochs,1);
    hist.loss  = zeros(cfg.epochs,1);
    hist.res   = zeros(cfg.epochs,1);
    hist.ic    = zeros(cfg.epochs,1);
    hist.data  = zeros(cfg.epochs,1);
    
    t0 = ic.t0;
    y0 = ic.y0;
    
    % optimizer flags
    use_adam = isfield(cfg,'optimizer') && strcmpi(cfg.optimizer,'adam');
    beta1 = cfg.beta1; beta2 = cfg.beta2; eps = cfg.eps;
    
    % learning rate (with optional scheduler)
    lr = cfg.lr;
    
    for ep = 1:cfg.epochs
    
        % LR scheduler: decay every 'decay_every' epochs by factor 'lr_decay'
        if ~isempty(cfg.lr_decay) && ~isempty(cfg.decay_every) && cfg.decay_every > 0
            if ep > 1 && mod(ep-1, cfg.decay_every) == 0
                lr = lr * cfg.lr_decay;
            end
        end
    
        % shuffle indices for mini-batches
        idx = randperm(Nc);
        ep_loss = 0; ep_res = 0; ep_ic = 0; ep_data = 0;
    
        for b = 1:num_batches
            j = idx( (b-1)*cfg.batch_size + 1 : min(b*cfg.batch_size, Nc) );
            t_batch = t_pool(:, j); % 1 x B (dlarray)
    
            % loss & grads via local function (dlfeval)
            [L, terms, grads] = dlfeval(@loss_and_grads_fn, ...
                net, sys, t0, y0, t_batch, cfg.loss_weights, data);
    
            % gradient clipping (global L2)
            if cfg.grad_clip > 0
                gnorm_sq = 0;
                for k = 1:numel(fn)
                    gnorm_sq = gnorm_sq + sum(grads.(fn{k}).^2,'all');
                end
                gnorm = sqrt(extractdata(gnorm_sq));
                if gnorm > cfg.grad_clip
                    scale = cfg.grad_clip / gnorm;
                    for k = 1:numel(fn)
                        grads.(fn{k}) = grads.(fn{k}) * scale;
                    end
                end
            end
    
            % parameter update
            if use_adam
                % Adam with bias correction (epoch-wise t = ep, batch-wise also ok)
                t_adam = ep; % simple counter; can be (ep-1)*num_batches+b
                for k = 1:numel(fn)
                    g = grads.(fn{k});
                    m.(fn{k}) = beta1 * m.(fn{k}) + (1 - beta1) * g;
                    v.(fn{k}) = beta2 * v.(fn{k}) + (1 - beta2) * (g.^2);
    
                    mhat = m.(fn{k}) ./ (1 - beta1^t_adam);
                    vhat = v.(fn{k}) ./ (1 - beta2^t_adam);
    
                    net.params.(fn{k}) = net.params.(fn{k}) - lr * mhat ./ (sqrt(vhat) + eps);
                end
            else
                % SGD + momentum
                for k = 1:numel(fn)
                    vtmp = cfg.momentum * vel.(fn{k}) - lr * grads.(fn{k});
                    vel.(fn{k}) = vtmp;
                    net.params.(fn{k}) = net.params.(fn{k}) + vtmp;
                end
            end
    
            % accumulate (host scalars)
            ep_loss = ep_loss + double(gather(extractdata(L)));
            ep_res  = ep_res  + double(gather(extractdata(terms.res)));
            ep_ic   = ep_ic   + double(gather(extractdata(terms.ic)));
            ep_data = ep_data + double(gather(extractdata(terms.data)));
        end
    
        hist.loss(ep)  = ep_loss / num_batches;
        hist.res(ep)   = ep_res  / num_batches;
        hist.ic(ep)    = ep_ic   / num_batches;
        hist.data(ep)  = ep_data / num_batches;

        print_every = 50;

        % metrics on full TRAIN data 
        do_metrics = isstruct(data) && ...
                     isfield(data,'t') && ...
                     isfield(data,'y') && ...
                     ~isempty(data) && ...
                     ~isempty(data.t) && ...
                     ~isempty(data.y);
        
        if do_metrics && (mod(ep, print_every) == 0 || ep == 1 || ep == cfg.epochs)
            Yhat = net.forward(data.t, net.params, 'eval');  % D x N (dlarray)
            Yhat = double(extractdata(Yhat));
            Yref = double(extractdata(data.y));
            M = compute_metrics(Yhat, Yref);
            hist.mse(ep)     = M.mse;
            hist.rmse(ep)    = M.rmse;
            hist.mae(ep)     = M.mae;
            hist.r2(ep)      = M.r2;
        end
    
        % logging
        if ep == 1 || mod(ep, print_every) == 0 || ep == cfg.epochs
            if do_metrics
                fprintf(['[%4d/%4d] lr=%.2e | loss=%.3e |' ...
                         ' MSE=%.3e, RMSE=%.3e, MAE=%.3f, R2=%.3f\n'], ...
                         ep, cfg.epochs, lr, hist.loss(ep), ...
                         hist.mse(ep), hist.rmse(ep), hist.mae(ep), hist.r2(ep));
                %fprintf('(res=%.3e, ic=%.3e, data=%.3e)', hist.res(ep), hist.ic(ep), hist.data(ep))
            else
                fprintf('[%4d/%4d] lr=%.2e | loss=%.3e | (res=%.3e, ic=%.3e, data=%.3e)\n', ...
                    ep, cfg.epochs, lr, hist.loss(ep), hist.res(ep), hist.ic(ep), hist.data(ep));
            end
        end
    end
end
 
%% ---- Utils ----

function [L, terms, grads] = loss_and_grads_fn(net, sys, t0, y0, t_batch, loss_weights, data)
    [L, terms] = losses(net, sys, t0, y0, t_batch, loss_weights, data);
    grads = dlgradient(L, net.params);
end

function M = compute_metrics(Yhat, Ytrue)
    % COMPUTE_METRICS  Overall metrics (flattened across dims/time).
    % Returns struct with fields: mse, rmse, mae, r2
    msk = ~isnan(Yhat) & ~isnan(Ytrue);
    yhat = Yhat(msk); y = Ytrue(msk);
    err = yhat - y;
    M.mse   = mean(err.^2);
    M.rmse  = sqrt(M.mse);
    M.mae   = mean(abs(err));
    ybar    = mean(y);
    sst     = sum((y - ybar).^2);
    sse     = sum(err.^2);
    M.r2    = 1 - sse / max(sst, eps);
end