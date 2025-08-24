function net = model(cfg)
    %% ---- Documentation ----

    % MODEL  Build a lightweight custom MLP for Physics-Informed Neural Nets (PINNs).
    %
    %   net = MODEL(cfg) returns a struct with:
    %     .cfg        original configuration
    %     .params     struct of learnable parameters as dlarray
    %     .forward    function handle: Y = net.forward(t, params, mode)
    %                  - t: 1xN dlarray (time samples, row vector)
    %                  - params: parameter struct (net.params by default)
    %                  - mode: 'train' (enables dropout) or 'eval'
    %                  Returns Y of size [D x N], where D = cfg.output_dim.
    %     .init       function handle to (re)initialize parameters
    %
    % Design goals:
    %   - No dlnetwork objects: pure dlarray workflow for full control.
    %   - Vectorized over columns (each column is a sample).
    
    %% ---- Model ----
 
    if nargin < 1 || isempty(cfg), cfg = struct(); end
    cfg = fill_defaults(cfg, struct( ...
        'input_dim',    1, ...
        'output_dim',   2, ...
        'hidden_sizes', [64 64 64], ...
        'activation',   'tanh', ...      % 'tanh' | 'relu' | 'swish'
        'dropout',      0.0, ...
        'dtype',        'single' ...     % 'single' | 'double'
    ));
    
    % basic validation
    if cfg.dropout < 0 || cfg.dropout > 1
        error('cfg.dropout must be in [0,1].');
    end
    if ~ismember(cfg.activation, {'tanh','relu','swish'})
        error('cfg.activation must be one of: tanh | relu | swish.');
    end
    if ~ismember(cfg.dtype, {'single','double'})
        error('cfg.dtype must be single or double.');
    end
    
    net.cfg = cfg;
    net.params = init_params(cfg, 0.1, cfg.dtype);  % Xavier-like with scale
    
    % forward(t, params, mode) -> D x N dlarray
    net.forward = @(t, params, mode) forward_mlp(t, params, cfg, mode);
    
    % init(scale) -> reset params with given scale
    net.init = @(scale) setfield(net, 'params', init_params(cfg, scale, cfg.dtype)); 
end
    
%% ---- utils ----
function cfg = fill_defaults(cfg, defaults)
    fn = fieldnames(defaults);

    for k = 1:numel(fn)
        f = fn{k};
        if ~isfield(cfg, f) || isempty(cfg.(f))
            cfg.(f) = defaults.(f);
        end
    end
end

function params = init_params(cfg, scale, dtype)
    sizes = [cfg.input_dim, cfg.hidden_sizes, cfg.output_dim];
    L = numel(sizes) - 1;

    for l = 1:L
        fan_in  = sizes(l);
        fan_out = sizes(l+1);
        limit = scale * sqrt(6/(fan_in + fan_out));   % Glorot uniform
        W = (2*rand(fan_out, fan_in, dtype) - 1) * limit;
        b = zeros(fan_out, 1, dtype);
        params.(sprintf('W%d', l)) = dlarray(W);
        params.(sprintf('b%d', l)) = dlarray(b);
    end
end

function Y = forward_mlp(t, params, cfg, mode)
    % t expected as 1xN row; we handle accidental column input by transposing.

    if ~isa(t,'dlarray'), t = dlarray(t); end
    if size(t,1) ~= cfg.input_dim && size(t,2) == cfg.input_dim
        t = t.';  % make it row
    end
    
    X = t;                  % (I x N), I = input_dim
    Lh = numel(cfg.hidden_sizes);
    A = X;
    
    % Hidden layers
    for l = 1:Lh
        W = params.(sprintf('W%d', l));
        b = params.(sprintf('b%d', l));
        Z = W*A + b;                          % (H_l x N)
        A = activate(Z, cfg.activation);
        
        if nargin >= 4 && strcmp(mode,'train') && cfg.dropout > 0
            pkeep = 1 - cfg.dropout;
            % generates random on the same device/type as the underlying array
            r = rand(size(A), 'like', extractdata(A)) < pkeep;  % logical numeric
            mask = dlarray(r);
            A = (A .* mask) / pkeep;  % inverted dropout
        end
    end
    
    % Output layer
    W = params.(sprintf('W%d', Lh+1));
    b = params.(sprintf('b%d', Lh+1));
    Y = W*A + b;                               % (D x N)
end
    
function Y = activate(Z, act)
    switch act
        case 'tanh',  Y = tanh(Z);
        case 'relu',  Y = max(Z, 0);
        case 'swish', Y = Z .* sigmoid(Z);
        otherwise,    error('Unknown activation %s', act);
    end
end
