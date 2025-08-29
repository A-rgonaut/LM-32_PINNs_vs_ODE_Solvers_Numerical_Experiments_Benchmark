function T = get_metrics(Yp_pinn, Yp_ode, Y_data)
% COMPARE_METRICS  Print and return a compact table comparing PINN vs ODE.
%   T = COMPARE_METRICS(Yp_pinn, Yp_ode, Y_data, label)
%   - Y* are D x N arrays (double)
%   - label (char/string) appears in the header (e.g., 'TEST' or 'TRAIN')
%
% Returns a MATLAB table with rows {'PINN','ODE'} and columns:
%   MSE, RMSE, MAE, R2

    Mp = metrics(Yp_pinn, Y_data);
    Mo = metrics(Yp_ode,  Y_data);

    rows = {'PINN'; 'ODE'};
    MSE  = [Mp.overall.mse;   Mo.overall.mse];
    RMSE = [Mp.overall.rmse;  Mo.overall.rmse];
    MAE  = [Mp.overall.mae;   Mo.overall.mae];
    R2   = [Mp.overall.r2;    Mo.overall.r2];

    T = table(MSE, RMSE, MAE, R2, 'RowNames', rows);

    fprintf('           MSE        RMSE        MAE        R2\n');
    fprintf('PINN  %10.3e  %10.3e  %10.3e  %8.3f \n', ...
        MSE(1), RMSE(1), MAE(1), R2(1));
    fprintf('ODE   %10.3e  %10.3e  %10.3e  %8.3f \n', ...
        MSE(2), RMSE(2), MAE(2), R2(2));
end

% ====================================================================== %

function M = metrics(Yhat, Ytrue)
% METRICS  Compute point-wise regression metrics for time-series/state data.
%   M = METRICS(Yhat, Ytrue)
%   Inputs:
%     Yhat  : D x N predictions
%     Ytrue : D x N reference (ground truth or dataset)
%   Output struct M with fields:
%     .per_dim.mse, .rmse, .mae, .r2   (1 x D)
%     .overall.mse, .rmse, .mae, .r2   (scalar)
%
% Notes:
%   - NaN in Ytrue/Yhat are ignored (masked) dimension-wise.
%   - R^2 is computed per-dimension using SST around the per-dim mean.
%   - relL2 = ||Yhat - Ytrue||_2 / ||Ytrue||_2 (per-dim and overall).
%   - pearson is Pearson correlation coefficient per state.

    arguments
        Yhat  double
        Ytrue double
    end

    [D, N] = size(Ytrue);
    assert(isequal(size(Yhat), [D, N]), 'Yhat and Ytrue must be D x N.');

    % mask NaNs dimension-wise
    mask = ~isnan(Ytrue) & ~isnan(Yhat);

    per_dim = struct('mse', [], 'rmse', [], 'mae', [], 'r2', []);
    per_dim.mse     = zeros(1,D);
    per_dim.rmse    = zeros(1,D);
    per_dim.mae     = zeros(1,D);
    per_dim.r2      = zeros(1,D);

    % per-dimension
    for d = 1:D
        msk = mask(d,:);
        y   = Ytrue(d, msk);
        yhat= Yhat(d,  msk);
        if isempty(y)
            per_dim.mse(d) = NaN; per_dim.rmse(d)=NaN; 
            per_dim.mae(d)=NaN; per_dim.r2(d) = NaN;  
            continue;
        end

        err = yhat - y;
        per_dim.mse(d)  = mean(err.^2);
        per_dim.rmse(d) = sqrt(per_dim.mse(d));
        per_dim.mae(d)  = mean(abs(err));

        % R^2
        ybar = mean(y);
        sst  = sum( (y - ybar).^2 );
        sse  = sum( err.^2 );
        per_dim.r2(d) = 1 - sse / max(sst, eps);  % avoid div by zero
        
    end

    % overall metrics (flatten across dims)
    msk_all = mask(:);
    y_all   = Ytrue(msk_all);
    yh_all  = Yhat(msk_all);
    err_all = yh_all - y_all;

    overall.mse   = mean(err_all.^2);
    overall.rmse  = sqrt(overall.mse);
    overall.mae   = mean(abs(err_all));
    ybar_all      = mean(y_all);
    sst_all       = sum( (y_all - ybar_all).^2 );
    sse_all       = sum( err_all.^2 );
    overall.r2    = 1 - sse_all / max(sst_all, eps);

    M = struct('per_dim', per_dim, 'overall', overall);
end