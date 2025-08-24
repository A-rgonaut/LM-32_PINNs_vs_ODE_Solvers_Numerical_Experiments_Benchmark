function sys = jerk(params)
    %% ---- Documentation ----
    
    % JERK  Third-order "jerk" differential equation in first-order form.
    %
    % Equation:
    %   y''' = -a y'' - y - (y')^2
    % State y = [y; y'; y''] gives:
    %   y1' = y2
    %   y2' = y3
    %   y3' = -a*y3 - y1 - y2.^2
    %
    % Parameters (defaults):
    %   a = 1.0
    %   tspan = [0 20]
    %   y0 = [1; 0; 0]
    %
    % Returns .f(t,y) vector field (D=3), no .acc (second-order only).
    
    %% ---- Jerk ----
    
    if nargin < 1 || isempty(params), params = struct(); end
    defaults = struct( ...
        'a',     1.0, ...
        'tspan', [0 20], ...
        'y0',    [1; 0; 0] ...
    );
    params = fill_defaults(params, defaults);
    
    sys.name      = 'Jerk (third order)';
    sys.order     = 3;
    sys.state_dim = 3;
    sys.tspan     = params.tspan;
    sys.y0        = params.y0(:);
    sys.param     = params;
    
    a = params.a;
    sys.f = @(t, y) rhs_jerk(t, y, a);
    
    sys.describe = 'Jerk system: y=[y; y''; y''],  y'''''' = -a y'''' - y - (y'')^2.';
end


%% ---- Utils ----
    
function dydt = rhs_jerk(~, y, a)
    dydt = zeros(3, size(y,2), 'like', y);
    dydt(1,:) = y(2,:);
    dydt(2,:) = y(3,:);
    dydt(3,:) = -a .* y(3,:) - y(1,:) - (y(2,:)).^2;
end

