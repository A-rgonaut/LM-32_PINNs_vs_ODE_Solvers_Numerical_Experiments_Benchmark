function init()
    %% ---- Documentation ----

    % INIT/DEMO  Train a PINN with real/synthetic dataset and compare
    % ODE vs PINN accuracy on the same data points.
    %
    % Dataset format (CSV) expected by default:
    %   t, y1, y2, ... , yD
    % where t is a column vector of times (N x 1) and y is N x D.
    %
    % If no CSV is provided, the script generates a synthetic dataset by
    % integrating the selected ODE and adding Gaussian noise.

    %% ---- Console ----

    % clear
    clc; 
    clear; 
    close all;
    
    % scoped path handling  
    oldPath = path; cleanupPath = onCleanup(@() path(oldPath));
    thisDir = fileparts(mfilename('fullpath')); addpath(genpath(thisDir));
    rehash toolboxcache

    %% ---- Solver -----
    
    % 'runge_kutta' | 'cranck_nicolson' | 'leapfrog' | ...
    which_solv  = "runge_kutta";        
    
    switch lower(which_solv)
        case 'euler_explicit',  solv = @euler_explicit;
        case 'euler_implicit',  solv = @euler_implicit;
        case 'crank_nicolson',  solv = @crank_nicolson;
        case 'leapfrog',        solv = @leapfrog;
        case 'runge_kutta',     solv = @runge_kutta;
        otherwise, error('Unknown solver "%s"', which_solv);
    end
    
    %% ---- System -----
    
    % 'hooke' | 'harmonic' | 'double_pendulum' | ...
    which_sys  = "blasius";        
    
    switch lower(which_sys)
        case 'hooke',           sys = hooke([]);
        case 'harmonic',        sys = harmonic([]);
        case 'double_pendulum', sys = double_pendulum([]);
        case 'van_der_pol',     sys = van_der_pol([]);
        case 'three_body',      sys = three_body([]);
        case 'blasius',         sys = blasius([]);
        case 'lorenz',          sys = lorenz([]);
        case 'jerk',            sys = jerk([]);
        otherwise, error('Unknown system "%s"', which_sys);
    end
    
    D = sys.state_dim;
    tspan = sys.tspan; 
    y0 = sys.y0; 

    %% ---- PINN ----

    % reproducibility
    rng(42);

    % PINN build config
    net_cfg = struct('input_dim',1,...
                     'output_dim',D,...
                     'hidden_sizes',[100 100 100], ...
                     'activation','relu', ...
                     'dropout',0.0, ...
                     'dtype','single');
    
    % PINN train config 
    train_cfg = struct();
    train_cfg.epochs        = 1500;
    train_cfg.batch_size    = 512;
    train_cfg.collocation_N = 4096;
    train_cfg.seed          = 42;
    train_cfg.lr            = 1e-2;
    train_cfg.momentum      = 0.9;  % 0.9
    train_cfg.grad_clip     = 5.0;  %5.0
    train_cfg.loss_weights  = struct('lambda_res',0.5,'lambda_ic',1,'lambda_data',0.5);
    train_cfg.optimizer     = 'sgd'; % 'sgd' | 'adam' 
    train_cfg.beta1         = 0.9;
    train_cfg.beta2         = 0.999;
    train_cfg.eps           = 1e-8;
    train_cfg.lr_decay      = [];   % scheduler
    train_cfg.decay_every   = [];   % ogni 50 epoche: lr *= 0.98

    %% ---- Dataset ----

    % settings
    % - csv_path, "" for synthetic
    % - noise_std, std for synthetic noise (ignored if csv provided)
    % - ode_step_h, step for synthetic ground-truth ODE  
    csv_path   = "lorenz_dataset_1.csv";
    noise_std  = 0; % 0.02 
    ode_step_h = 1e-3;

    % split (train 90%, test 10%) — chronological mode for time series
    train_ratio = 0.90;           
    split_mode  = "chronological"; 
    
    % load or synthesize dataset
    if strlength(csv_path) > 0
        T = readmatrix(csv_path);
        if size(T,2) ~= D+1
            error('CSV must have 1 time column + %d state columns. Found %d.', D, size(T,2)-1);
        end
        t_all = T(:,1).';
        y_all = T(:,2:end).';
    else
        % synthetic: integrate ODE on a sparser grid inside [t0 t1], then add noise
        t_dense = tspan(1):ode_step_h:tspan(2);
        Y_dense = solv(sys.f, t_dense, y0);   % D x Ndense

        % pick an observation grid (coarser)
        Nobs = 200;
        ii = round(linspace(1, numel(t_dense), Nobs));
        t_all = t_dense(ii);
        y_all = Y_dense(:, ii);

        % add noise
        y_all = y_all + noise_std * randn(size(y_all));
        t_all = t_all(:).';     % row 1xN
    end
    
    % sort by time (robust for per ODE and chronological split)
    [t_all, order] = sort(t_all(:).');
    y_all = y_all(:, order);
    
    % train/test split (90/10, cronologico) ----
    [train_set, test_set] = split_train_test(t_all, y_all, train_ratio, split_mode);

    % prepare TRAIN data for PINN 
    data_train = struct();
    data_train.t = dlarray(train_set.t);   % 1 x Ntr
    data_train.y = dlarray(train_set.y);   % D x Ntr

    %% ---- Training and Testing PINN ----

    % build PINN 
    net = model(net_cfg);
    
    % train PINN with TRAIN data
    ic = struct('t0', tspan(1), 'y0', y0);
    [net, hist] = train(net, sys, tspan, ic, train_cfg, data_train);
    
    % PINN's predictions on TRAIN and TEST times 
    Yp_pinn_train = net.forward(dlarray(train_set.t), net.params, 'eval');
    Yp_pinn_test  = net.forward(dlarray(test_set.t),  net.params, 'eval');

    Yp_pinn_train = double(extractdata(Yp_pinn_train));   % D x Ntr
    Yp_pinn_test  = double(extractdata(Yp_pinn_test));    % D x Nte
    
    % ODE's solutions on TRAIN and TEST times 
    Yp_ode_train = solv(sys.f, train_set.t, y0);   % D x Ntr
    Yp_ode_test  = solv(sys.f, test_set.t,  y0);   % D x Nte
    
    % metrics 
    fprintf('\nSystem: (%s)\n', sys.name);

    fprintf('\n== Metrics TRAIN (%.0f%%) ==\n', 100*(train_ratio));
    train_metrics = get_metrics(Yp_pinn_train, Yp_ode_train, train_set.y);
    fprintf('--> %s wins \n', ternary(train_metrics{'PINN','MSE'}, train_metrics{'ODE','MSE'}));

    fprintf('\n== Metrics TEST (%.0f%%) ==\n', 100*(1-train_ratio));
    test_metrics = get_metrics(Yp_pinn_test, Yp_ode_test, test_set.y);
    fprintf('---> %s wins \n', ternary(test_metrics{'PINN','MSE'}, test_metrics{'ODE','MSE'}));

    %% ---- Results ----
    
    % plot: dataset vs ODE vs PINN
    figure('Color','w'); 
    tiledlayout(max(2, min(3,D)), 1, 'TileSpacing','compact');
    K = min(D, 3);  % plot max 3 panels
    
    for d = 1:K
        nexttile; hold on; grid on;
        
        % data
        plot(train_set.t, train_set.y(d,:), '.', ...
            'DisplayName', sprintf('data train  y_%d', d));
        plot(test_set.t,  test_set.y(d,:),  '.', ...
            'DisplayName', sprintf('data test   y_%d', d));
        
        % ODE
        plot(train_set.t, Yp_ode_train(d,:), '-',  ...
            'DisplayName', sprintf('ODE train  y_%d', d));
        plot(test_set.t,  Yp_ode_test(d,:),  '-',  ...
            'DisplayName', sprintf('ODE test   y_%d', d));
        
        % PINN
        plot(train_set.t, Yp_pinn_train(d,:), '--', ...
            'LineWidth',1.5, ...
            'DisplayName', sprintf('PINN train y_%d', d));
        plot(test_set.t,  Yp_pinn_test(d,:),  '--', ...
            'LineWidth',1.5, ...
            'DisplayName', sprintf('PINN test  y_%d', d));
    
        xlabel('t'); ylabel(sprintf('state %d', d));
        
        if d==1
            title(sprintf('%s: dataset vs ODE vs PINN (split 90/10)', sys.name)); 
        end
        
        legend('Location','bestoutside');
    end
    
    % plot training loss
    figure('Color','w');
    plot(hist.epoch, hist.loss, 'LineWidth', 1.5); grid on;
    xlabel('epoch'); ylabel('loss'); title('PINN training loss (TRAIN set)');
    
    % summary 
    fprintf('\n== Summary ==\n');
    fprintf('  System: %s\n', sys.name);
    fprintf('  Points: Ntrain = %d, Ntest = %d, state dim = %d\n', ...
            numel(train_set.t), ...
            numel(test_set.t), ...
            D);
    fprintf('  Loss weights: res=%g, ic=%g, data=%g\n', ...
            train_cfg.loss_weights.lambda_res, ...
            train_cfg.loss_weights.lambda_ic, ...
            train_cfg.loss_weights.lambda_data);
    fprintf('  MSE(PINN)=%.3e  MSE(ODE)=%.3e  --> %s wins.\n\n', ...
            test_metrics{'PINN','MSE'}, ...
            test_metrics{'ODE','MSE'}, ...
            ternary(test_metrics{'PINN','MSE'}, test_metrics{'ODE','MSE'}));
end
    
%% ---- Utils ----

function [train_set, test_set] = split_train_test(t, y, train_ratio, mode)
    % split_train_test  (chronological) return struct with params t, y
    N = numel(t);
    Ntr = floor(N * train_ratio);
    Ntr = min(max(Ntr, 1), N-1); % almeno 1 test
    
    switch lower(mode)
        case 'chronological'
            idx_tr = 1:Ntr;
            idx_te = Ntr+1:N;
        otherwise
            error('Unsupported split mode: %s', mode);
    end
    
    train_set.t = t(idx_tr);
    train_set.y = y(:, idx_tr);
    
    test_set.t  = t(idx_te);
    test_set.y  = y(:, idx_te);
end

function out = ternary(a,b)
    if a < b, out = 'PINN'; else, out = 'ODE'; end
end