function [acc_mv] = run_EM(f, g, h)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[M, N] = size(f);
K = max(f, [], 'all');
estm_g = zeros(N, 1);
estm_h = zeros(N, 1);
for j = 1 : N
    count = zeros(K, 1);
    for i = 1 : M
        if f(i, j) ~= 0
            count(f(i, j)) = count(f(i, j)) + 1;
        end
    end
    [~, order] = sort(count, 'descend');
    estm_g(j) = order(1);
    estm_h(j) = order(2);
end
acc_mv(1) = nnz(g == estm_g);
acc_mv(2) = nnz(h == estm_h);
acc_mv(3) = nnz((g == estm_g) & (h == estm_h));
end

