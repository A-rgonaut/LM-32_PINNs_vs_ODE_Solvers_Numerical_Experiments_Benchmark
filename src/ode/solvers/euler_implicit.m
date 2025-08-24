function Y = euler_implicit(f, t, y0, tol, maxit)
% EULER_IMPLICIT  Backward Euler for y' = f(t,y) using Newton iterations.
%   Y = euler_implicit(f, t, y0, tol, maxit)
% Inputs:
%   f     : @(t,y) D x 1 vector field
%   t     : 1xN time grid (uniform or not)
%   y0    : D x 1 initial state
%   tol   : stopping tolerance on update (default 1e-8)
%   maxit : max Newton iterations per step (default 20)
if nargin < 4 || isempty(tol), tol = 1e-8; end
if nargin < 5 || isempty(maxit), maxit = 20; end

N = numel(t);
D = numel(y0);
Y = zeros(D, N);
Y(:,1) = y0;

for n = 1:N-1
    h = t(n+1) - t(n);
    tn1 = t(n+1);
    % initial guess: explicit Euler
    y = Y(:,n) + h * f(t(n), Y(:,n));
    for it = 1:maxit
        F = y - Y(:,n) - h * f(tn1, y);
        if norm(F,2) < tol, break; end
        J = jacobian(@(z) z - Y(:,n) - h * f(tn1, z), y);
        dy = -J \ F;
        y = y + dy;
        if norm(dy,2) < tol, break; end
    end
    Y(:,n+1) = y;
end
end

function J = jacobian(F, y)
% Finite-difference Jacobian of F at y
D = numel(y);
J = zeros(D,D);
eps0 = 1e-6 * max(1, norm(y,2));
for i = 1:D
    e = zeros(D,1); e(i) = 1;
    h = eps0 * (1 + abs(y(i)));
    yph = y + h*e;
    ymh = y - h*e;
    J(:,i) = (F(yph) - F(ymh)) / (2*h);
end
end
