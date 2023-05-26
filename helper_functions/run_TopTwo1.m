function [acc_toptwo1, estm_p, estm_q] = run_TopTwo1(f, g, h)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Y_obs = f;
[N, M] = size(Y_obs);
K = max(Y_obs, [], 'all');
S = nnz(Y_obs) / numel(Y_obs);

shifted_matrix= zeros(K, N, M);
us = zeros(K-1, N);

for k = 1 : K-1
    observed_matrix_bin = Y_obs;
    observed_matrix_bin((1 <= Y_obs) & (Y_obs <= k)) = -1;
    observed_matrix_bin(Y_obs>k) = 1;
    observed_matrix_bin = observed_matrix_bin - S * (K-2*k)/K;

    [U, ~, ~] = svd(observed_matrix_bin);
    if sum(U(:, 1)) >0
        us(k, :) = U(:, 1);
    else
        us(k, :) = -U(:, 1);
    end

    shifted_matrix(k, :, :) = observed_matrix_bin;
end

avg_u = mean(us, 1);
avg_u = max(min(avg_u, 1), 0);

w = zeros(M, K+1);
for k = 1 : K-1
    w(:, k+1) = squeeze(shifted_matrix(k, :, :))'*avg_u' / (2*S);
end

del_w = zeros(M, K);
for k = 1 : K
    del_w(:, k) = w(:, k+1) - w(:, k);
end

[estm_g, estm_h] = top2(-del_w);

acc_toptwo1(1) = nnz(g == estm_g);
acc_toptwo1(2) = nnz(h == estm_h);
acc_toptwo1(3) = nnz((g == estm_g) & (h == estm_h));

estm_p = sum(Y_obs == repmat(estm_g', N, 1) | Y_obs == repmat(estm_h', N, 1), 2);
worker_degree = sum(Y_obs > 0, 2);

estm_p = estm_p ./ worker_degree;
estm_p = estm_p - 2/K;
estm_p = estm_p / (1-2/K);

estm_p = max(min(estm_p, 0.999), 0.001);
norm_p = norm(estm_p);

estm_q = ones(M, 1) / K;
for j = 1 : M
    estm_q(j) = estm_q(j) - (del_w(j, g(j)) / norm_p);
end

estm_q = max(min(estm_q, 0.999), 0.501);

end

