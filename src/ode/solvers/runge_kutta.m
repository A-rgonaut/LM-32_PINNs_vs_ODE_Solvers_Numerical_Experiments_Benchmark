function Y = runge_kutta(f, t, y0)
% RUNGE_KUTTA  Classic RK4 for y' = f(t,y).
N = numel(t);
D = numel(y0);
Y = zeros(D, N);
Y(:,1) = y0;
for n = 1:N-1
    h = t(n+1) - t(n);
    tn = t(n); yn = Y(:,n);
    k1 = f(tn, yn);
    k2 = f(tn + h/2, yn + h/2 * k1);
    k3 = f(tn + h/2, yn + h/2 * k2);
    k4 = f(tn + h,   yn + h   * k3);
    Y(:,n+1) = yn + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end
end
