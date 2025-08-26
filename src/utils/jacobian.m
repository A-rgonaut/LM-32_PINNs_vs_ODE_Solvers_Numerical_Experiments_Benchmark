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