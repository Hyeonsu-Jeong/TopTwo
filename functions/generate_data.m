function [observed_matrix, ground_truth, most_confusing_answer] = generate_data(worker_reliability, task_difficulty, K)

[N, ~] = size(worker_reliability);
[M, ~] = size(task_difficulty);
observed_matrix = zeros(N, M);
ground_truth = randi(K, M, 1);

most_confusing_answer = randi(K-1, M, 1);
most_confusing_answer(ground_truth == most_confusing_answer) = most_confusing_answer(ground_truth == most_confusing_answer) + 1;

for i = 1 : N
    for j = 1 : M
        prob_list = 0:K;
        prob_list = prob_list * (1-worker_reliability(i));
        prob_list = prob_list / K;
        prob_list(ground_truth(j)+1:end) = prob_list(ground_truth(j)+1:end) + worker_reliability(i) * task_difficulty(j);
        prob_list(most_confusing_answer(j)+1:end) = prob_list(most_confusing_answer(j)+1:end) + worker_reliability(i) * (1-task_difficulty(j));
        rv = rand();
        for k = 1 : K
            if prob_list(k) <= rv && rv < prob_list(k+1)
                observed_matrix(i, j) = k;
            end
        end
    end
end

