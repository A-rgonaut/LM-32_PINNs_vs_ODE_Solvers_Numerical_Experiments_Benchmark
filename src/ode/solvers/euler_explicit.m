function Y = euler_explicit(f, t, y0)
    %% ---- Documentation ----

    % EULER_IMPLICIT  Classic Euler for y' = f(t,y).
    %   Y = euler_implicit(f, t, y0, tol, maxit)
    % 
    % Inputs:
    %   f     : @(t,y) D x 1 vector field
    %   t     : 1xN time grid (uniform or not)
    %   y0    : D x 1 initial state
    %   tol   : stopping tolerance on update (default 1e-8)
    %   maxit : max Newton iterations per step (default 20)
    %
    % Outpus:
    %   Y: D x N trajectory.
    
    %% ---- Euler Explicit ----

    h = t(2) - t(1);
    N = numel(t);
    D = numel(y0);
    Y = zeros(D, N);
    Y(:,1) = y0;

    for n = 1:N-1
        Y(:,n+1) = Y(:,n) + h * f(t(n), Y(:,n));
    end
end
