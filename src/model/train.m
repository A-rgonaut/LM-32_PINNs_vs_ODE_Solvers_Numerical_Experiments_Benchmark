function [net, hist] = train(net, sys, tspan, ic, cfg, data)
    %% ---- Documentation ----

    % TRAIN  PINN training loop (mini-batch, momentum, gradient clipping).
    %
    % [net, hist] = TRAIN(net, sys, [t0 t1], ic, cfg, data)
    %
    % ic: struct with fields .t0 (scalar), .y0 (D x 1)
    %
    % cfg fields:
    %   .epochs        (default 2000)
    %   .batch_size    (default 128)
    %   .collocation_N (default 4096)
    %   .seed          (default 42)
    %   .lr            (default 1e-3)
    %   .momentum      (default 0.9)
    %   .grad_clip     (default 5.0; 0 disables)
    %   .loss_weights  (struct with fields: lambda_res=1, lambda_ic=1, lambda_data=0)

    %% ---- Train ----

    if nargin < 6, data = []; end
    if nargin < 5, cfg = struct(); end
    
    cfg = fill_defaults(cfg, struct( ...
        'epochs',        2000, ...
        'batch_size',    128, ...
        'collocation_N', 4096, ...
        'seed',          42, ...
        'lr',            1e-3, ...
        'momentum',      0.9, ...
        'grad_clip',     5.0, ...
        'loss_weights',  struct('lambda_res',1,'lambda_ic',1,'lambda_data',0) ...
    ));
    
    % ---- collocation pool ----
    rng(cfg.seed);
    t_pool = collocation(tspan, cfg.collocation_N, 'random'); % 1 x Nc (dlarray)
    Nc = size(t_pool,2);
    
    % ---- momentum buffers ----
    vel = struct();
    fn = fieldnames(net.params);

    for k = 1:numel(fn)
        vel.(fn{k}) = zeros(size(net.params.(fn{k})), 'like', net.params.(fn{k}));
    end

    num_batches = ceil(Nc / cfg.batch_size);
    hist.epoch = zeros(cfg.epochs,1);
    hist.loss  = zeros(cfg.epochs,1);
    hist.res   = zeros(cfg.epochs,1);
    hist.ic    = zeros(cfg.epochs,1);
    hist.data  = zeros(cfg.epochs,1);
    
    t0 = ic.t0;
    y0 = ic.y0;

    for ep = 1:cfg.epochs
        idx = randperm(Nc);
        ep_loss = 0; ep_res=0; ep_ic=0; ep_data=0;
    
        for b = 1:num_batches
            j = idx( (b-1)*cfg.batch_size + 1 : min(b*cfg.batch_size, Nc) );
            t_batch = t_pool(:, j); % 1 x B (dlarray)
    
            % closure for loss & grads (must run under dlfeval)
            [L, terms, grads] = dlfeval(@loss_and_grads_fn, ...
                net, sys, t0, y0, t_batch, cfg.loss_weights, data);
    
            % gradient clipping (global L2)
            if cfg.grad_clip > 0
                gnorm_sq = 0;
                for k = 1:numel(fn)
                    g = grads.(fn{k});
                    gnorm_sq = gnorm_sq + sum(g.^2,'all');
                end
                gnorm = sqrt(extractdata(gnorm_sq));
                if gnorm > cfg.grad_clip
                    scale = cfg.grad_clip / gnorm;
                    for k = 1:numel(fn)
                        grads.(fn{k}) = grads.(fn{k}) * scale;
                    end
                end
            end
    
            % SGD + momentum
            for k = 1:numel(fn)
                v = cfg.momentum * vel.(fn{k}) - cfg.lr * grads.(fn{k});
                vel.(fn{k}) = v;
                net.params.(fn{k}) = net.params.(fn{k}) + v;
            end
    
            % accumulate
            ep_loss = ep_loss + double(gather(extractdata(L)));
            ep_res  = ep_res  + double(gather(extractdata(terms.res)));
            ep_ic   = ep_ic   + double(gather(extractdata(terms.ic)));
            ep_data = ep_data + double(gather(extractdata(terms.data)));
        end
    
        hist.epoch(ep) = ep;
        hist.loss(ep)  = ep_loss / num_batches;
        hist.res(ep)   = ep_res  / num_batches;
        hist.ic(ep)    = ep_ic   / num_batches;
        hist.data(ep)  = ep_data / num_batches;
    
        if mod(ep, max(1,ceil(cfg.epochs/10))) == 0 || ep == 1
            fprintf('[%4d/%4d] loss=%.3e  (res=%.3e, ic=%.3e, data=%.3e)\n', ...
                ep, cfg.epochs, hist.loss(ep), hist.res(ep), hist.ic(ep), hist.data(ep));
        end
    end
end

%% ---- Utils ----

function [L, terms, grads] = loss_and_grads_fn(net, sys, t0, y0, t_batch, loss_weights, data)
    [L, terms] = losses(net, sys, t0, y0, t_batch, loss_weights, data);
    grads = dlgradient(L, net.params);
end

function cfg = fill_defaults(cfg, defaults)
    fn = fieldnames(defaults);
    for k = 1:numel(fn)
        f = fn{k};
        if ~isfield(cfg, f) || isempty(cfg.(f))
            cfg.(f) = defaults.(f);
        end
    end
end
