function [tgrid, Ypred] = evaluate(net, tspan, N)
    %% ---- Documentation ----

    % EVALUATE  Evaluate PINN on a uniform grid for plotting.
    % [tgrid, Ypred] = EVALUATE(net, [t0 t1], N)
    %   returns tgrid (1xN double), Ypred (D x N double).
 
    %% ---- Evaluate ----

    if nargin < 3, N = 1001; end
    
    tgrid = linspace(tspan(1), tspan(2), N);
    Y = net.forward(dlarray(tgrid), net.params, 'eval'); % dlarray D x N
    Ypred = gather(extractdata(Y));
end
