function Y = euler_explicit(f, t, y0)
% EULER_EXPLICIT  Classic forward Euler on y' = f(t,y).
h = t(2) - t(1);
N = numel(t);
D = numel(y0);
Y = zeros(D, N);
Y(:,1) = y0;
for n = 1:N-1
    Y(:,n+1) = Y(:,n) + h * f(t(n), Y(:,n));
end
end
