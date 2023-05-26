function [F,f,F_orig,f_orig,g,h,K,M,N,conf_mat,p,q] = LoadDataset(sampling_matrix, ground_truth, most_confusing_answer, worker_reliability, task_difficulty)
 % parameter
    valid_worker = [];
    valid_task = [];

    [total_worker, total_task] = size(sampling_matrix);
    for row = 1:total_worker
        if sum(sampling_matrix(row, :)) > 0
            valid_worker = [valid_worker, row];
        end
    end
    for col = 1:total_task
        if sum(sampling_matrix(:,col)) > 0
            valid_task= [valid_task, col];
        end
    end

    f = sampling_matrix(valid_worker, valid_task);
    M = size(valid_worker, 2);
    N = size(valid_task, 2);
    g = ground_truth(valid_task);
    h = most_confusing_answer(valid_task);
    p = worker_reliability(valid_worker);
    q = task_difficulty(valid_task);
    K = max(sampling_matrix, [], 'all');
    f_orig = f;

    conf_mat = ones(K, N, M);
    

    F = cell(M,1); %cell of annotator responses. 
    F_orig = cell(M,1);
    for i=1:M 
        indx = find(f(i,:) > 0);
        F{i} = sparse(f(i,indx),indx,1,K,N);
    end
    for i=1:M 
        indx = find(f_orig(i,:) > 0);
        F_orig{i} = sparse(f_orig(i,indx),indx,1,K,N);
    end
end