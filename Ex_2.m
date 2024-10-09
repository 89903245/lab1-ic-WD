clc;
clear all;

m = 128; 
n = 256; 
S = 12;  

rng(2);  

A = randn(m, n);
A = normc(A);  

x = zeros(n, 1);
spars_index = randsample(n, S);
x(spars_index) = randn(S, 1);  

y = A * x;  

% OMP 
x_hat_omp = OMP(A, y, S);

% IHT 
x_hat_iht = IHT(A, y, S, 100, 1e-6);

% SP 
x_hat_sp = SubspacePursuit(A, y, S, 20);

%
figure;
subplot(4,2,1);
plot(x, 'b');
title('Original Signal');

% OMP
subplot(4,2,3);
plot(x_hat_omp, 'b');
title('Recovered Signal by OMP');
subplot(4,2,4);
plot(x_hat_omp - x, 'b'); 
title('Difference (OMP)');

% IHT
subplot(4,2,5);
plot(x_hat_iht, 'm');
title('Recovered Signal by IHT');
subplot(4,2,6);
plot(x_hat_iht - x, 'm');  
title('Difference (IHT)');

% SP
subplot(4,2,7);
plot(x_hat_sp, 'c');
title('Recovered Signal by SP');
subplot(4,2,8);
plot(x_hat_sp - x, 'c');  
title('Difference (SP)');

% IHT 
function x_hat = IHT(A, y, S, maxIter, tol)
    x_hat = zeros(size(A, 2), 1);
    for i = 1:maxIter
        x_hat = x_hat + A' * (y - A * x_hat);
        [~, idx] = sort(abs(x_hat), 'descend');
        x_hat(idx(S+1:end)) = 0; 
        if norm(y - A * x_hat) < tol
            break;
        end
    end
end

% Subspace Pursuit 
function x_hat = SubspacePursuit(A, y, S, maxIter)
    T = [];
    x_hat = zeros(size(A, 2), 1);
    r = y;
    for iter = 1:maxIter
        T = union(T, find(abs(A' * r) > 0.1*max(abs(A' * r))));
        if length(T) > 2*S
            [~, idx] = sort(abs(A(:, T)' * y), 'descend');
            T = T(idx(1:2*S));
        end
        x_T = A(:, T) \ y;
        x_hat(T) = x_T;
        r = y - A * x_hat;
        if norm(r) < 1e-6
            break;
        end
    end
    [~, idx] = sort(abs(x_hat), 'descend');
    x_hat(idx(S+1:end)) = 0; 
end

% OMP 
function x_hat = OMP(A, y, S)
    x_hat = zeros(size(A, 2), 1);
    y_r = y;
    S_set = [];

    for k = 1:S
        [~, ind] = max(abs(A' * y_r));
        if ~ismember(ind, S_set)
            S_set = [S_set, ind];
            A_S = A(:, S_set);
            x_S = pinv(A_S) * y;
            x_hat(S_set) = x_S;
            y_r = y - A * x_hat;
        end
    end
end