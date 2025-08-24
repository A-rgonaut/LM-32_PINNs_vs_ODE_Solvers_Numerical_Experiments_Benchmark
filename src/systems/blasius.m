function sys = blasius(params)
% BLASIUS  Blasius boundary-layer ODE (third order) as a first-order system.
%
% Original ODE (eta is the independent variable):
%   f''' + 0.5 f f'' = 0,
%   f(0) = 0,  f'(0) = 0,  f'(inf) = 1.
%
% We set y = [f; f'; f''], then:
%   y1' = y2
%   y2' = y3
%   y3' = -0.5 * y1 * y3
%
% We treat it as an IVP using "shooting": choose s = f''(0) and integrate
% on [0, etamax]. The default s ~= 0.4696 yields f'(etamax) ~ 1 for
% etamax ~ 8..10.
%
% sys fields:
%   .name, .order=3, .state_dim=3, .tspan=[0,etamax], .y0=[0;0;s],
%   .param = struct('s',..., 'etamax', ...), .f(t,y)

if nargin < 1 || isempty(params), params = struct(); end
defaults = struct( ...
    's',      0.4696, ...   % initial guess for f''(0)
    'etamax', 10.0   ...    % finite truncation of [0,inf)
);
params = fill_defaults(params, defaults);

sys.name      = 'Blasius boundary layer';
sys.order     = 3;
sys.state_dim = 3;
sys.tspan     = [0, params.etamax];
sys.y0        = [0; 0; params.s];   % [f(0); f''(0) = 0; f''(0)=s]
sys.param     = params;

sys.f = @(t, y) rhs_blasius(t, y);

sys.describe = 'Blasius: y=[f; f''; f''],  y'''' = -0.5 f f''. Shooting with f''''(0)=s.';
end

% ---- helpers ----
function dydt = rhs_blasius(~, y)
% y = [f; fp; fpp]
dydt = zeros(3, size(y,2), 'like', y);
dydt(1,:) = y(2,:);              % f'   = fp
dydt(2,:) = y(3,:);              % fp'  = fpp
dydt(3,:) = -0.5 .* y(1,:) .* y(3,:);  % fpp' = -0.5 f fpp
end

function s = fill_defaults(s, d)
fn = fieldnames(d);
for k = 1:numel(fn)
    f = fn{k};
    if ~isfield(s, f) || isempty(s.(f)), s.(f) = d.(f); end
end
end
