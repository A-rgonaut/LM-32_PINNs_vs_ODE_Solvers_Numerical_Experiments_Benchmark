function sys = double_pendulum(params)
% DOUBLE_PENDULUM  Planar double pendulum with angles theta1, theta2.
%
% State vector (first-order form):
%   y = [theta1; omega1; theta2; omega2]
%
% Parameters (defaults shown):
%   m1, m2 : masses
%   l1, l2 : lengths
%   g      : gravity
%
% Equations adapted from standard references; vectorized over columns.
if nargin < 1 || isempty(params)
    params.m1 = 1.0; params.m2 = 1.0;
    params.l1 = 1.0; params.l2 = 1.0;
    params.g  = 9.81;
end
m1 = params.m1; m2 = params.m2;
l1 = params.l1; l2 = params.l2;
g  = params.g;

sys.name      = 'Double pendulum';
sys.order     = 2;
sys.state_dim = 4;
sys.tspan     = [0, 20];
sys.y0        = [pi/2; 0; pi/2; 0];
sys.param     = params;

sys.f   = @(t, y) double_pendulum_rhs(t, y, m1, m2, l1, l2, g);
% For leapfrog: x=[th1; th2], v=[w1; w2] -> acc: [th1dd; th2dd]
sys.acc = @(t, x, v) double_pendulum_acc(t, x, v, m1, m2, l1, l2, g);

sys.describe = 'Double pendulum with y=[th1; w1; th2; w2].';
end

function dydt = double_pendulum_rhs(~, y, m1, m2, l1, l2, g)
th1 = y(1,:); w1 = y(2,:);
th2 = y(3,:); w2 = y(4,:);
d = th2 - th1;

den1 = (m1 + m2) * l1 - m2 * l1 .* cos(d).^2;
den2 = (l2 ./ l1) .* den1;

th1dot = w1;
th2dot = w2;

w1dot = ( m2 * l1 .* w1.^2 .* sin(d) .* cos(d) ...
        + m2 * g * sin(th2) .* cos(d) ...
        + m2 * l2 .* w2.^2 .* sin(d) ...
        - (m1 + m2) * g * sin(th1) ) ./ den1;

w2dot = ( -m2 * l2 .* w2.^2 .* sin(d) .* cos(d) ...
        + (m1 + m2) * ( g * sin(th1) .* cos(d) ...
        - l1 * w1.^2 .* sin(d) - g * sin(th2) ) ) ./ den2;

dydt = [th1dot; w1dot; th2dot; w2dot];
end

function acc = double_pendulum_acc(t, x, v, m1, m2, l1, l2, g)
% Returns [th1dd; th2dd] given x=[th1; th2], v=[w1; w2]
y = [x(1,:); v(1,:); x(2,:); v(2,:)];
dydt = double_pendulum_rhs(t, y, m1, m2, l1, l2, g);
acc = [dydt(2,:); dydt(4,:)];
end
