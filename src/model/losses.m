function [L, terms] = losses(net, sys, t0, y0, t_col, cfg, data)
    %% ---- Documentation ----

    % LOSSES  Compute PINN loss terms.
    %
    % [L, terms] = LOSSES(net, sys, t0, y0, t_col, cfg, data)
    %   t0, y0 : initial condition (scalar t0, column y0 [D x 1])
    %   t_col  : 1xNc collocation points (dlarray row)
    %   cfg    : struct with weights:
    %              .lambda_res (default 1)
    %              .lambda_ic  (default 1)
    %              .lambda_data(default 0)
    %   data   : optional struct with fields .t (1xNd), .y (D x Nd)
    %
    % Returns:
    %   L      : scalar dlarray loss
    %   terms  : struct with fields res, ic, data
    
    %% ---- Losses ----
 
    % defaults
    if nargin < 6 || isempty(cfg), cfg = struct; end
    if ~isfield(cfg,'lambda_res'),  cfg.lambda_res  = 1.0; end
    if ~isfield(cfg,'lambda_ic'),   cfg.lambda_ic   = 1.0; end
    if ~isfield(cfg,'lambda_data'), cfg.lambda_data = 0.0; end
    
    % Residual term on collocation batch
    [R, ~, ~] = residual(net, sys, t_col, 'train');
    L_res = mean(R.^2, 'all');
    
    % Initial condition term
    t0_dl = dlarray(reshape(t0,1,[]));
    Y0 = net.forward(t0_dl, net.params, 'eval');   % D x 1
    L_ic = mean((Y0 - y0).^2, 'all');
    
    % Optional data term
    if nargin >= 7 && ~isempty(data) && isfield(data,'t') && isfield(data,'y') && ~isempty(data.t)
        Yd = net.forward(data.t, net.params, 'eval'); % D x Nd
        L_data = mean((Yd - data.y).^2, 'all');
    else
        L_data = L_res*0;
    end
    
    L = cfg.lambda_res * L_res + cfg.lambda_ic * L_ic + cfg.lambda_data * L_data;
    
    terms = struct('res', L_res, 'ic', L_ic, 'data', L_data);
end
