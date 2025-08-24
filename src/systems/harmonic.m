function sys = harmonic(params)
% HARMONIC  Damped harmonic oscillator: m x'' + c x' + k x = 0
%
% State y = [x; v], first-order form:
%   x' = v
%   v' = -(k/m) x - (c/m) v
if nargin < 1 || isempty(params)
    params.m = 1.0;
    params.c = 0.2;
    params.k = 1.0;
end
m = params.m; c = params.c; k = params.k;

sys.name      = 'Damped harmonic oscillator';
sys.order     = 2;
sys.state_dim = 2;
sys.tspan     = [0, 20];
sys.y0        = [1; 0];
sys.param     = params;

sys.f   = @(t, y) [y(2,:); -(k/m) * y(1,:) - (c/m) * y(2,:)];
sys.acc = @(t, x, v) -(k/m) * x - (c/m) * v;

sys.describe = 'Damped harmonic oscillator y=[x; v] with x'''' = -(k/m) x - (c/m) v.';
end
