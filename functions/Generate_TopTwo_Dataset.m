function [observed_matrix, ground_truth, most_confusing_answer] = Generate_TopTwo_Dataset(total_worker, total_task, total_label, worker_reliability, task_difficulty)

observed_matrix = zeros(total_worker, total_task);
ground_truth = randi(total_label, total_task, 1);

most_confusing_answer = randi(total_label-1, total_task, 1);
most_confusing_answer(ground_truth == most_confusing_answer) = most_confusing_answer(ground_truth == most_confusing_answer) + 1;

for i = 1 : total_worker
    for j = 1 : total_task
        prob_list = 0:total_label;
        prob_list = prob_list * (1-worker_reliability(i));
        prob_list = prob_list / total_label;
        prob_list(ground_truth(j)+1:end) = prob_list(ground_truth(j)+1:end) + worker_reliability(i) * task_difficulty(j);
        prob_list(most_confusing_answer(j)+1:end) = prob_list(most_confusing_answer(j)+1:end) + worker_reliability(i) * (1-task_difficulty(j));
        rv = rand();
        for k = 1 : total_label
            if prob_list(k) <= rv && rv < prob_list(k+1)
                observed_matrix(i, j) = k;
            end
        end
    end
end


