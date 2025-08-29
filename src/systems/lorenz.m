function sys = lorenz(params)
    %% ---- Documentation ----
    
    % LORENZ  Classic Lorenz attractor system.
    %   dx/dt = sigma*(y - x)
    %   dy/dt = x*(rho - z) - y
    %   dz/dt = x*y - beta*z
    %
    % USO:
    %   sys = lorenz([])                      % default params
    %   sys = lorenz(struct('sigma',10,...))  % override
    %
    % sys fields:
    %   .name       = 'lorenz'
    %   .state_dim  = 3
    %   .tspan      = [0, 40]        (default)
    %   .y0         = [1; 1; 1]      (default)
    %   .params     = struct with sigma, rho, beta
    %   .f(t,u)     = RHS handle (3x1)
    %   .jac(t,u)   = Jacobian handle (3x3)  [opzionale]

    %% ---- Lorenz ----
    
    if nargin < 1 || isempty(params), params = struct(); end
    defaults = struct( ...
        'sigma',     10, ...
        'rho',       28, ...
        'beta',      8/3, ...
        'tspan',     [0 40], ...
        'y0',        [1; 1; 1].'); % y0 = [1; 1; 1]
    params = fill_defaults(params, defaults);

    sys.name      = 'Lorenz';
    sys.state_dim = 3;
    sys.tspan     = params.tspan;
    sys.y0        = params.y0;
    sys.params    = params;

    % RHS
    sys.f = @(t,u) lorenz_rhs(t,u,params);
end

%% ---- Utils ----    

function du = lorenz_rhs(~, u, p)
    x = u(1); y = u(2); z = u(3);
    du = [ p.sigma*(y - x);
           x*(p.rho - z) - y;
           x*y - p.beta*z ];
end