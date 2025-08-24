function [R, dYdt, Y] = residual(net, sys, t, mode)
    %% ---- Documentation ----
    % RESIDUAL  Physics residual for first-order system y' = f(t,y).
    % 
    % [R, dYdt, Y] = RESIDUAL(net, sys, t, mode)
    %   - net: struct from model()
    %   - sys: system struct with .f(t,y) returning D x N
    %   - t  : 1xN dlarray of time samples (row)
    %   - mode: 'train' or 'eval' (affects dropout in net.forward)
    % Returns:
    %   Y    : D x N predicted states
    %   dYdt : D x N time derivatives (via dlgradient)
    %   R    : D x N residuals dYdt - f(t,Y)
    %
    % Implementation detail:
    %   We compute dYdt for each output channel i by differentiating the scalar
    %   sum(Y(i,:)) wrt t (vector). This yields a 1xN gradient with entries
    %   dY(i,j)/dt_j under the assumption that sample j depends only on t_j,
    %   which holds for our vectorized, column-wise network.
     
    %% ---- Residual ----
    
    if nargin < 4, mode = 'eval'; end
    if ~isa(t,'dlarray'), t = dlarray(t); end
    
    % Forward pass
    Y = net.forward(t, net.params, mode);  % D x N
    
    % Compute derivative dYdt
    D = size(Y,1);
    N = size(Y,2);
    dYdt = dlarray(zeros(D, N, 'like', Y));
    
    % IMPORTANT: Do gradients inside dlfeval caller. Here we assume residual()
    % is called from within a dlfeval context (which train.m/losses.m ensure).
    for i = 1:D
        s = sum(Y(i,:), 'all');                 % scalar dlarray
        gi = dlgradient(s, t, 'EnableHigherDerivatives', true);  % 1 x N
        dYdt(i,:) = gi;
    end
    
    % System RHS
    fY = sys.f(t, Y);   % D x N
    R = dYdt - fY;      % D x N
end
