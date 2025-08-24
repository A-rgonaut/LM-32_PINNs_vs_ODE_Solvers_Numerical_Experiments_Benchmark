function t_col = collocation(tspan, N, mode, seed)
% COLLOCATION  Generate collocation time points.
%
% t_col = COLLOCATION([t0 t1], N, mode, seed)
%   mode: 'random' (default) or 'grid'
%   returns a 1xN dlarray (row).
if nargin < 3 || isempty(mode), mode = 'random'; end
if nargin >= 4 && ~isempty(seed), rng(seed); end
t0 = tspan(1); t1 = tspan(2);
switch lower(mode)
    case 'random'
        t = t0 + (t1 - t0) * rand(1, N, 'single');
        % Ensure t0 is included occasionally for good IC coverage
        t(1) = t0;
    case 'grid'
        t = linspace(t0, t1, N);
    otherwise
        error('Unknown mode %s', mode);
end
t_col = dlarray(t);
end
