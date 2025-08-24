function sys = hooke(params)
    %% ---- Documentation ----
    
    % HOOKE  Linear spring-mass system: m x'' + k x = 0
    % 
    % Returns a system struct usable by ODE solvers and the PINN:
    %   .name         (char)
    %   .order        (1 or 2; here: 2)
    %   .state_dim    (2)  [x; v]
    %   .tspan        default time span
    %   .y0           default initial state
    %   .param        parameter struct with fields m, k
    %   .f(t, y)      first-order system y' = f(t,y)
    %   .acc(t, x, v) acceleration for leapfrog (optional)
    %
    % y = [x; v]
    
    %% ---- Hooke ----
    
    if nargin < 1 || isempty(params)
        params.m = 1.0;
        params.k = 1.0;
    end
    m = params.m; k = params.k;
    
    sys.name      = 'Hooke';
    sys.order     = 2;
    sys.state_dim = 2;
    sys.tspan     = [0, 10];
    sys.y0        = [1; 0];
    sys.param     = params;
    
    sys.f   = @(t, y) [y(2,:); -(k/m) * y(1,:)];
    sys.acc = @(t, x, v) -(k/m) * x;
    
    sys.describe = 'Linear spring-mass oscillator y=[x; v] with x'''' = -(k/m) x.';
end
