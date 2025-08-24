function sys = three_body(params)
    %% ---- Documentation ----
    
    % THREE_BODY  Planar Newtonian 3-body problem (2D) with softening.
    %
    % State (first-order form, D=12):
    %   y = [r1; v1; r2; v2; r3; v3],      each r_i = [x_i; y_i], v_i = [u_i; w_i]
    % Dynamics:
    %   r_i' = v_i
    %   v_i' = sum_{j!=i} G*m_j*(r_j - r_i) / (||r_j - r_i||^2 + eps^2)^(3/2)
    %
    % Returns a system struct compatible with the rest of the project:
    %   .name, .order=2, .state_dim=12, .tspan, .y0, .param, .f(t,y), .acc(t,x,v)
    %
    % PARAMETERS (fields of 'params', all optional; defaults shown):
    %   G       : 1.0          % gravitational constant (nondimensional)
    %   m       : [1 1 1]      % masses [m1 m2 m3]
    %   eps     : 1e-3         % softening length (stability near close encounters)
    %   r0      : 2x3          % initial positions as columns [r1 r2 r3]
    %   v0      : 2x3          % initial velocities as columns [v1 v2 v3]
    %   tspan   : [0 20]       % time interval
    %
    % Notes:
    %   - Default IC are chosen for a visually interesting, non-trivial motion
    %     and to avoid immediate singularities. Adjust r0/v0 as you like.
    %   - Use .acc with leapfrog/Verlet (expects x=[r1;r2;r3] (6xN), v=[v1;v2;v3] (6xN)).

    %% ---- Three Body ----
    
    % -------- defaults --------
    if nargin < 1 || isempty(params), params = struct(); end
    defaults = struct( ...
        'G',     1.0, ...
        'm',     [1 1 1], ...
        'eps',   1e-3, ...
        'tspan', [0 20], ...
        'r0',    [-1 0   1  0   0  0.8].', ...  % r1=[-1;0], r2=[1;0], r3=[0;0.8]
        'v0',    [ 0 0.25  0 -0.25  -0.30 0].'); % v1=[0;0.25], v2=[0;-0.25], v3=[-0.30;0]
    params = fill_defaults(params, defaults);
    
    % reshape r0,v0 if provided as vector
    if isvector(params.r0), params.r0 = reshape(params.r0, [2,3]); end
    if isvector(params.v0), params.v0 = reshape(params.v0, [2,3]); end
    
    % pack initial state y0 = [r1; v1; r2; v2; r3; v3] (12x1)
    y0 = [params.r0(:,1); params.v0(:,1); ...
          params.r0(:,2); params.v0(:,2); ...
          params.r0(:,3); params.v0(:,3)];
    
    % -------- system struct --------
    sys.name      = 'Three-body (planar)';
    sys.order     = 2;
    sys.state_dim = 12;
    sys.tspan     = params.tspan;
    sys.y0        = y0;
    sys.param     = params;
    
    G   = params.G;
    m   = params.m(:).';   % row [m1 m2 m3]
    eps = params.eps;
    
    sys.f   = @(t, y) rhs_three_body(t, y, G, m, eps);
    sys.acc = @(t, x, v) acc_three_body(t, x, v, G, m, eps); % x=[r1;r2;r3], v=[v1;v2;v3]
    
    sys.describe = 'Planar Newtonian 3-body with softening; y=[r1;v1;r2;v2;r3;v3], D=12.';
end
    
%% ---- Utils ----    

function dydt = rhs_three_body(~, y, G, m, eps)
% y: 12xN -> split into r1,v1,r2,v2,r3,v3 (each 2xN)
N  = size(y, 2);
r1 = y(1:2,    :); v1 = y(3:4,    :);
r2 = y(5:6,    :); v2 = y(7:8,    :);
r3 = y(9:10,   :); v3 = y(11:12,  :);

[a1, a2, a3] = pairwise_acc(r1, r2, r3, G, m, eps);

dydt = zeros(12, N, 'like', y);
dydt(1:2,   :) = v1;  dydt(3:4,   :) = a1;
dydt(5:6,   :) = v2;  dydt(7:8,   :) = a2;
dydt(9:10,  :) = v3;  dydt(11:12, :) = a3;
end

function A = acc_three_body(~, x, v, G, m, eps) %#ok<INUSD>
    % x: 6xN = [r1; r2; r3], v unused here (no drag); return A=[a1; a2; a3] (6xN)
    r1 = x(1:2,   :);
    r2 = x(3:4,   :);
    r3 = x(5:6,   :);
    [a1, a2, a3] = pairwise_acc(r1, r2, r3, G, m, eps);
    A = [a1; a2; a3];
end

function [a1, a2, a3] = pairwise_acc(r1, r2, r3, G, m, eps)
    % Vectorized accelerations for each body (columns are samples).
    % a_i = sum_{j!=i} G*m_j*(r_j - r_i) / (||r_j - r_i||^2 + eps^2)^(3/2)
    dr12 = r2 - r1;  dr13 = r3 - r1;
    dr21 = -dr12;    dr23 = r3 - r2;
    dr31 = -dr13;    dr32 = -dr23;
    
    d12 = sqrt(sum(dr12.^2,1) + eps^2);
    d13 = sqrt(sum(dr13.^2,1) + eps^2);
    d23 = sqrt(sum(dr23.^2,1) + eps^2);
    
    a1 = G * ( m(2) * dr12 ./ (d12.^3) + m(3) * dr13 ./ (d13.^3) );
    a2 = G * ( m(1) * dr21 ./ (d12.^3) + m(3) * dr23 ./ (d23.^3) );
    a3 = G * ( m(1) * dr31 ./ (d13.^3) + m(2) * dr32 ./ (d23.^3) );
end