function [acc_ebcc] = run_TopTwo1(f, g, h)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [N, M] = size(f);
    K = max(f, [], 'all');
    tuples = zeros(nnz(f), 3);
    
    a_pi=0.1;
    alpha=1;
    a_v=4;
    b_v=1; 
    seed=1234;
    max_iter=500;

    tuple_idx = 1;
    for worker=1:N
        for task=1:M
            if f(worker, task) ~= 0
                tuples(tuple_idx, :) = [task, worker, f(worker, task)];
            end
            
        end
        tuple_idx = tuple_idx + 1;
    end
    y_is_one_lij = [];
    y_is_one_lji = [];

    for k=1:K
        selected = find(tuples(:, 3) == k);
        coo_ij = sparse(tuples(selected, 1), tuples(selected, 2), ones([size(selected), 1]), M, N)
        y_is_one_lij = [y_is_one_lij, coo_ij];
        y_is_one_lji = [y_is_one_lji, coo_ij];

    beta_kl = eye(K) * (a_v - b_v) + b_v

    final_z_ik = zeros(M, K);
    final_ELBO = -Inf

    for iter=1:10
        z_ik = zeros(M, K);
        for l=1:K
            z_ik(:, l) = z_ik(:, l) + sum(y_is_one_lij(l), 2)
        end
        z_ik = z_ik ./ sum(z_ik, 2)
    
end

