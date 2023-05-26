function [acc_toptwo2] = run_TopTwo2(f, g, h, p, q)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Y_obs = f;
[N, M] = size(Y_obs);
K = max(Y_obs, [], 'all');
S = nnz(Y_obs) / numel(Y_obs);

estm_g = zeros(M, 1);
estm_h = zeros(M, 1);

for j = 1 : M
    first_coeff = log(K * q(j) * (p./(1-p)) + 1);
    second_coeff = log(K *(1-q(j)) * (p./(1-p)) + 1);

    sum_g = zeros(K, 1);
    sum_h = zeros(K, 1);

    for i = 1 : N
        if Y_obs(i, j) ~= 0
            sum_g(Y_obs(i, j)) = sum_g(Y_obs(i, j)) + first_coeff(i);
            sum_h(Y_obs(i, j)) = sum_h(Y_obs(i, j)) + second_coeff(i);
        end
    end

    max_val = -1;
    for a = 1 : K
        for b = 1 : K
            if a ~= b && max_val < sum_g(a) + sum_h(b)
                max_val =  sum_g(a) + sum_h(b);
                estm_g(j) = a;
                estm_h(j) = b;
            end
        end
    end
end

acc_toptwo2(1) = nnz(g == estm_g);
acc_toptwo2(2) = nnz(h == estm_h);
acc_toptwo2(3) = nnz((g == estm_g) & (h == estm_h));
end

