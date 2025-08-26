function sols = run_solvers(sys, tspan, y0, h)
    %% ---- Documentation ----

    % RUN_SOLVERS  Run the suite of ODE solvers on a given system.
    %
    % sols = run_solvers(sys, tspan, y0, h) returns a struct with fields:
    %   .t                 common time grid
    %   .euler_explicit
    %   .euler_implicit
    %   .crank_nicolson
    %   .leapfrog          (only if system looks second-order [x; v])
    %   .runge_kutta
    %
    % Each field is a DxN state trajectory.
    
    %% ---- All solvers runner ----

    t = tspan(1):h:tspan(2);
    f = sys.f;
    D = numel(y0);
    
    sols.t = t;
    sols.euler_explicit = euler_explicit(f, t, y0);
    sols.euler_implicit = euler_implicit(f, t, y0);
    sols.crank_nicolson = crank_nicolson(f, t, y0);
    sols.runge_kutta    = runge_kutta(f, t, y0);
    
    % Leapfrog only for systems that look like [x; v] and provide acceleration
    if isfield(sys,'acc') && sys.order == 2 && D >= 2 && mod(D,2) == 0
        try
            sols.leapfrog = leapfrog(@(tt, x, v) sys.acc(tt, x, v), t, y0);
        catch E
            warning('leapfrog failed: %s', warning(E.message));
            sols.leapfrog = NaN(D, numel(t));
        end
    else
        sols.leapfrog = NaN(D, numel(t));
    end
end
