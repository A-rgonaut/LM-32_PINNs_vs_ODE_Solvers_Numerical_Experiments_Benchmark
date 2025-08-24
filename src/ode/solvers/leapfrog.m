function Y = leapfrog(accel, t, y0)
% LEAPFROG  Velocity-Verlet/Störmer-Verlet for second-order systems.
%   Assumes state y = [x; v], possibly with x and v vectors.
%   accel: @(t, x, v) returns a(x,v,t) with same shape as x.
%
% Y = leapfrog(accel, t, y0)
%   t: 1xN time grid
%   y0: D x 1 initial state where D = 2*d (x and v stacked)
%   Returns D x N trajectory.
N = numel(t);
D = numel(y0);
d = D/2;
if mod(D,2) ~= 0
    error('leapfrog expects an even-sized state y=[x; v].');
end

Y = zeros(D, N);
x = y0(1:d,1);
v = y0(d+1:end,1);
Y(:,1) = y0;

for n = 1:N-1
    h = t(n+1) - t(n);
    a_n = accel(t(n), x, v);
    v_half = v + 0.5*h * a_n;
    x_new  = x + h * v_half;
    a_np1  = accel(t(n+1), x_new, v_half); % allow velocity-dependent accel
    v_new  = v_half + 0.5*h * a_np1;
    x = x_new; v = v_new;
    Y(:,n+1) = [x; v];
end
end
