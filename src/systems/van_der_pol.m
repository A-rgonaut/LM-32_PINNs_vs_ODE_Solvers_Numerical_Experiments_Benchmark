function sys = van_der_pol(params)
    %% ---- Documentation ----
    
    % VANDERPOL  Van der Pol oscillator: x'' - mu (1 - x^2) x' + x = 0
    %
    % First-order system with y = [x; v]:
    %   x' = v
    %   v' = mu (1 - x^2) v - x
    
    %% ---- Van Der Pol ----
    
    if nargin < 1 || isempty(params)
        params.mu = 3.0;
    end
    mu = params.mu;
    
    sys.name      = 'Van der Pol';
    sys.order     = 2;
    sys.state_dim = 2;
    sys.tspan     = [0, 20];
    sys.y0        = [1; 0];
    sys.param     = params;
    
    sys.f   = @(t, y) [y(2,:); mu * (1 - y(1,:).^2) .* y(2,:) - y(1,:)];
    sys.acc = @(t, x, v) mu * (1 - x.^2) .* v - x;
    
    sys.describe = 'Van der Pol oscillator y=[x; v] with x'''' = mu(1-x^2)x'' - x.';
end
