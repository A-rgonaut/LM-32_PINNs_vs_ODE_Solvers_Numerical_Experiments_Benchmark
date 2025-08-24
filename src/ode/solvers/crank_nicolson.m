function Y = crank_nicolson(f, t, y0, tol, maxit)
% CRANK_NICOLSON  Trapezoidal rule (implicit midpoint) for y' = f(t,y).
%   Solves  y_{n+1} = y_n + h/2 ( f(t_n, y_n) + f(t_{n+1}, y_{n+1}) )
if nargin < 4 || isempty(tol), tol = 1e-8; end
if nargin < 5 || isempty(maxit), maxit = 20; end

N = numel(t);
D = numel(y0);
Y = zeros(D, N);
Y(:,1) = y0;

for n = 1:N-1
    h = t(n+1) - t(n);
    tn = t(n); tn1 = t(n+1);
    fn = f(tn, Y(:,n));
    % initial guess: explicit trapezoid
    y = Y(:,n) + h * fn;
    for it = 1:maxit
        F = y - Y(:,n) - (h/2) * ( fn + f(tn1, y) );
        if norm(F,2) < tol, break; end
        J = eye(D) - (h/2) * jacobian(@(z) f(tn1, z), y);
        dy = -J \ F;
        y = y + dy;
        if norm(dy,2) < tol, break; end
    end
    Y(:,n+1) = y;
end
end

function J = jacobian(F, y)
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
