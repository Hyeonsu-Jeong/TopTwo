% clc
% clear
% close
% addpath('functions')
% 
% total_worker = 50;
% total_task = 500;
% total_label = 5;
% 
% q_min = 0.501;
% q_max = 0.999;
% 
% p_mean_list = 0.25:0.05:0.75;
% max_iter = 3;
% 
% % generate task difficulty
% error_mv = zeros(size(p_mean_list, 2), max_iter, 3);
% error_alg1 = zeros(size(p_mean_list, 2), max_iter, 3);
% error_alg2 = zeros(size(p_mean_list, 2), max_iter, 3);
% error_oracle = zeros(size(p_mean_list, 2), max_iter, 3);
% error_mv_em = zeros(size(p_mean_list, 2), max_iter, 3);
% error_sm_em = zeros(size(p_mean_list, 2), max_iter, 3);
% error_pgd = zeros(size(p_mean_list, 2), max_iter, 3);
% 
% for p_idx = 1 : size(p_mean_list, 2)
%     p_mean = p_mean_list(p_idx);
%     p_min = p_mean - 0.249;
%     p_max = p_mean + 0.249;
%     worker_reliability = (p_max - p_min) * rand(total_worker, 1) + p_min;
%     task_difficulty = (q_max - q_min) * rand(total_task, 1) + q_min;
%     
%     [observed_matrix, ground_truth, most_confusing_answer] = generate_data(worker_reliability, task_difficulty, total_label);
%     
%     for times = 1:max_iter
%         fprintf("%d %d\n", p_idx, times);
%         Y_obs = zeros(total_worker, total_task);
%         for i = 1 : total_worker
%             for j = 1 : total_task
%                 if rand() < 0.2
%                     Y_obs(i, j) = observed_matrix(i, j);
%                 end
%             end
%         end
%         
%         trimmed_ground_truth = ground_truth;
%         trimmed_ground_truth(~any(Y_obs, 1)) = [];
%         
%         trimmed_most_confusing_answer = most_confusing_answer;
%         trimmed_most_confusing_answer(~any(Y_obs, 1)) = [];
%         
%         trimmed_worker_reliability = worker_reliability;
%         
%         trimmed_task_difficulty= task_difficulty;
%         trimmed_task_difficulty(~any(Y_obs, 1)) = [];
%         Y_obs(:, ~any(Y_obs, 1)) = [];
%         
%         [num_worker, num_task] = size(Y_obs);
%         num_class = total_label;
%         
%         Y_obs_separate = zeros(num_worker, num_task, num_class);
%         for i = 1 : num_class
%             separate1 = zeros(num_worker, num_task);
%             separate1(find(Y_obs == i)) = 1;
%             Y_obs_separate(:, :, i) = separate1;
%         end
%         
%         N = zeros(num_worker);
%         for i = 1 : num_worker
%             for j = 1 : num_worker
%                 if i == j
%                     N(i, j) = 0;
%                 else
%                     N(i, j) = sum(Y_obs(i, :) & Y_obs(j, :));
%                 end
%             end
%         end
%         
%         C = zeros(num_worker);
%         for i = 1 : num_worker
%             for j = 1 : num_worker
%                 if N(i, j) ~= 0
%                     valid_idx = Y_obs(i, :) & Y_obs(j, :);
%                     C(i, j) = num_class/((num_class - 1) * N(i,j)) * sum((Y_obs(i, :) ...
%                         == Y_obs(j, :)).* valid_idx) - 1/(num_class - 1); % multilabel equation in JMLR
%                 end
%             end
%         end
%         
%         y = trimmed_ground_truth;
%         A = [];
%         Z = zeros(num_task, num_class, num_worker);
%         for i = 1 : num_class
%             index = find(Y_obs_separate(:,:,i) ~= 0);
%             [worker_idx , task_idx] = ind2sub([num_worker, num_task], index);
%             class_idx = i * ones(size(task_idx));
%             A = [A; task_idx, worker_idx, class_idx];
%             index_Z = sub2ind(size(Z), task_idx, class_idx, worker_idx);
%             Z(index_Z) = 1;
%         end
%         
%         n = num_task;
%         m = num_worker;
%         k = num_class;
%         
%         Nround = 1;
%         mode = 1;
%         
%         error1_predict = zeros(6, Nround);
%         error2_predict = zeros(6, Nround);
%         
%         
%         % ====================================================
%         % n = max(A(:,1));
%         % m = max(A(:,2));
%         % k = max(B(:,2));
%         %
%         % y = zeros(n,1);
%         % for i = 1:size(B,1)
%         %     y(B(i,1)) = B(i,2);
%         % end
%         valid_index = find(y > 0);
%         %
%         % Z = zeros(n,k,m);
%         % for i = 1:size(A,1)
%         %     Z(A(i,1),A(i,3),A(i,2)) = 1;
%         % end
%         %===================== EM with majority vote ================
%         q = mean(Z,3);
%         q = q ./ repmat(sum(q,2),1,k);
%         mu = zeros(k,k,m);
%         
%         % EM update
%         for i = 1:m
%             mu(:,:,i) = (Z(:,:,i))'*q;
%             mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
%             
%             for c = 1:k
%                 mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
%             end
%         end
%         
%         q = zeros(n,k);
%         for j = 1:n
%             for c = 1:k
%                 for i = 1:m
%                     if Z(j,:,i)*mu(:,c,i) > 0
%                         q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
%                     end
%                 end
%             end
%             q(j,:) = exp(q(j,:));
%             q(j,:) = q(j,:) / sum(q(j,:));
%         end
%         
%         [g, h] = top2(q);
%         error_mv_em(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%         % ===================== EM with spectral method ==============
%         % method of moment
%         group = mod(1:m,3)+1;
%         Zg = zeros(n,k,3);
%         cfg = zeros(k,k,3);
%         for i = 1:3
%             I = find(group == i);
%             Zg(:,:,i) = sum(Z(:,:,I),3);
%         end
%         
%         x1 = Zg(:,:,1)';
%         x2 = Zg(:,:,2)';
%         x3 = Zg(:,:,3)';
%         
%         muWg = zeros(k,k+1,3);
%         muWg(:,:,1) = SolveCFG(x2,x3,x1);
%         muWg(:,:,2) = SolveCFG(x3,x1,x2);
%         muWg(:,:,3) = SolveCFG(x1,x2,x3);
%         
%         mu = zeros(k,k,m);
%         for i = 1:m
%             x = Z(:,:,i)';
%             x_alt = sum(Zg,3)' - Zg(:,:,group(i))';
%             muW_alt = (sum(muWg,3) - muWg(:,:,group(i)));
%             mu(:,:,i) = (x*x_alt'/n) / (diag(muW_alt(:,k+1)/2)*muW_alt(:,1:k)');
%             
%             mu(:,:,i) = max( mu(:,:,i), 10^-6 );
%             mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
%             for j = 1:k
%                 mu(:,j,i) = mu(:,j,i) / sum(mu(:,j,i));
%             end
%         end
%         
%         % EM update
%         
%         q = zeros(n,k);
%         for j = 1:n
%             for c = 1:k
%                 for i = 1:m
%                     if Z(j,:,i)*mu(:,c,i) > 0
%                         q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
%                     end
%                 end
%             end
%             q(j,:) = exp(q(j,:));
%             q(j,:) = q(j,:) / sum(q(j,:));
%         end
%         
%         for i = 1:m
%             mu(:,:,i) = (Z(:,:,i))'*q;
%             
%             mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
%             for c = 1:k
%                 mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
%             end
%         end
%         
%         [g, h] = top2(q);
%         error_sm_em(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%         %====================== PGD ==========================================
%         x_p = 0.5*ones(num_worker,1);
%         alpha=1e-5;
%         x1 = zeros(num_worker,1);
%         t = 0;
%         while sum(abs(x_p-x1))>1e-10
%             x1=x_p;
%             x_p = x_p + alpha*grad1(x_p,abs(C),N);
%             x_p=min(x_p,1-1./sqrt(num_task));
%             x_p=max(x_p,-1/(num_class-1)+1./sqrt(num_task));
%             sum(abs(x_p-x1));
%             t = t + 1;
%             if t == 300000
%                 break
%             end
%         end
%         probWorker = x_p.*(num_class-1)/(num_class)+1/num_class; % x is s in JMLR, which is not probability
%         predlabel = zeros(num_task,1);
%         Error = 0;
%         weights = log(probWorker.*(num_class-1)./(1-probWorker));
%         score = zeros(num_task, num_class);
%         for i = 1 : num_class
%             score(:, i) = sum(Y_obs_separate(:, :, i) .* repmat(weights, 1, num_task), 1);
%         end
%         [g, h] = top2(score);
%         error_pgd(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%         % ================================ TopTwo1 ================================
%         [N, M] = size(Y_obs);
%         K = max(Y_obs, [], 'all');
%         S = nnz(Y_obs) / numel(Y_obs);
%         
%         shifted_matrix= zeros(K, N, M);
%         us = zeros(K-1, N);
%         
%         for k = 1 : K-1
%             observed_matrix_bin = Y_obs;
%             observed_matrix_bin((1 <= Y_obs) & (Y_obs <= k)) = -1;
%             observed_matrix_bin(Y_obs>k) = 1;
%             observed_matrix_bin = observed_matrix_bin - S * (K-2*k)/K;
%             
%             [U, ~, ~] = svd(observed_matrix_bin);
%             if sum(U(:, 1)) >0
%                 us(k, :) = U(:, 1);
%             else
%                 us(k, :) = -U(:, 1);
%             end
%             
%             shifted_matrix(k, :, :) = observed_matrix_bin;
%         end
%         
%         avg_u = mean(us, 1);
%         avg_u = max(min(avg_u, 1), 0);
%         
%         w = zeros(M, K+1);
%         for k = 1 : K-1
%             w(:, k+1) = squeeze(shifted_matrix(k, :, :))'*avg_u' / (2*S);
%         end
%         
%         del_w = zeros(M, K);
%         for k = 1 : K
%             del_w(:, k) = w(:, k+1) - w(:, k);
%         end
%         
%         [g, h] = top2(-del_w);
%         error_alg1(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%         % ================================ TopTwo2 ================================
%         
%         
%         p = sum(Y_obs == repmat(g', N, 1) | Y_obs == repmat(h', N, 1), 2);
%         worker_degree = sum(Y_obs > 0, 2);
%         
%         p = p ./ worker_degree;
%         p = p - 2/K;
%         p = p / (1-2/K);
%         
%         p = max(min(p, 0.999), 0.001);
%         norm_p = norm(p);
%         
%         q = ones(M, 1) / K;
%         for j = 1 : M
%             q(j) = q(j) - (del_w(j, g(j)) / norm_p);
%         end
%         
%         q = max(min(q, 0.999), 0.501);
%         % q = 0.498 * (q - min(q))/(max(q)-min(q)) + 0.501;
%         
%         g = zeros(M, 1);
%         h = zeros(M, 1);
%         
%         for j = 1 : M
%             first_coeff = log(K * q(j) * (p./(1-p)) + 1);
%             second_coeff = log(K *(1-q(j)) * (p./(1-p)) + 1);
%             
%             sum_g = zeros(K, 1);
%             sum_h = zeros(K, 1);
%             
%             for i = 1 : N
%                 if Y_obs(i, j) ~= 0
%                     sum_g(Y_obs(i, j)) = sum_g(Y_obs(i, j)) + first_coeff(i);
%                     sum_h(Y_obs(i, j)) = sum_h(Y_obs(i, j)) + second_coeff(i);
%                 end
%             end
%             
%             max_val = -1;
%             for a = 1 : K
%                 for b = 1 : K
%                     if a ~= b && max_val < sum_g(a) + sum_h(b)
%                         max_val =  sum_g(a) + sum_h(b);
%                         g(j) = a;
%                         h(j) = b;
%                     end
%                 end
%             end
%         end
%         error_alg2(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%         % ================================ Oracle ================================
%         p = trimmed_worker_reliability;
%         q = trimmed_task_difficulty;
%         
%         g = zeros(M, 1);
%         h = zeros(M, 1);
%         
%         for j = 1 : M
%             first_coeff = log(K * q(j) * (p./(1-p)) + 1);
%             second_coeff = log(K *(1-q(j)) * (p./(1-p)) + 1);
%             
%             sum_g = zeros(K, 1);
%             sum_h = zeros(K, 1);
%             
%             for i = 1 : N
%                 if Y_obs(i, j) ~= 0
%                     sum_g(Y_obs(i, j)) = sum_g(Y_obs(i, j)) + first_coeff(i);
%                     sum_h(Y_obs(i, j)) = sum_h(Y_obs(i, j)) + second_coeff(i);
%                 end
%             end
%             
%             max_val = -1;
%             for a = 1 : K
%                 for b = 1 : K
%                     if a ~= b && max_val < sum_g(a) + sum_h(b)
%                         max_val =  sum_g(a) + sum_h(b);
%                         g(j) = a;
%                         h(j) = b;
%                     end
%                 end
%             end
%         end
%         error_oracle(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%         % ================================ MV =====================================
%         
%         g = zeros(M, 1);
%         h = zeros(M, 1);
%         for j = 1 : M
%             count = zeros(K, 1);
%             for i = 1 : N
%                 if Y_obs(i, j) ~= 0
%                     count(Y_obs(i, j)) = count(Y_obs(i, j)) + 1;
%                 end
%             end
%             [~, order] = sort(count, 'descend');
%             g(j) = order(1);
%             h(j) = order(2);
%         end
%         error_mv(p_idx, times, :) = 1 - grade(trimmed_ground_truth, trimmed_most_confusing_answer, g, h) / total_task;
%     end
% end

mean_mv = squeeze(mean(error_mv, 2))';
std_mv = squeeze(std(error_mv, 0, 2))';

mean_alg1 = squeeze(mean(error_alg1, 2))';
std_alg1 = squeeze(std(error_alg1, 0, 2))';

mean_alg2 = squeeze(mean(error_alg2, 2))';
std_alg2 = squeeze(std(error_alg2, 0, 2))';

mean_oracle = squeeze(mean(error_oracle, 2))';
std_oracle = squeeze(std(error_oracle, 0, 2))';

mean_mv_em = squeeze(mean(error_mv_em, 2))';
std_mv_em = squeeze(std(error_mv_em, 0, 2))';

mean_sm_em = squeeze(mean(error_sm_em, 2))';
std_sm_em = squeeze(std(error_sm_em, 0, 2))';

mean_pgd = squeeze(mean(error_pgd, 2))';
std_pgd = squeeze(std(error_pgd, 0, 2))';

color_list = {[0.4940 0.1840 0.5560], [0.8500 0.3250 0.0980], [0.3010 0.7450 0.9330], [0.4660 0.6740 0.1880], [1 0 0], [0 0 1], [0 0 0]};

title_list = {'P($(g, h) \neq (\hat{g}, \hat{h}$))', 'P($g \neq \hat{g}$)', 'P($h \neq \hat{h}$)'};
marker_list = {'p', 'o',  '<', 'x','s','^', 'd'};
fill_color_list = color_list;

x =  p_mean_list;

fig = figure(1);
set(fig, 'OuterPosition', [100, 200, 2000, 600])

for plot_idx = 1 : 3
    subplot(1, 3, plot_idx);
    error_mean = [mean_mv(plot_idx, :); mean_mv_em(plot_idx, :); mean_sm_em(plot_idx, :); mean_pgd(plot_idx, :); mean_alg1(plot_idx, :); mean_alg2(plot_idx, :); mean_oracle(plot_idx, :)];
    error_std = [std_mv(plot_idx, :); std_mv_em(plot_idx, :); std_sm_em(plot_idx, :); std_pgd(plot_idx, :); std_alg1(plot_idx, :); std_alg2(plot_idx, :); std_oracle(plot_idx, :)];
    
    error_lb = error_mean - error_std;
    error_ub = error_mean + error_std;
    for i = 1 : 7
        A(i) = plot(x, error_mean(i, :), 'Color', color_list{i}, 'LineStyle', '-',...
            'LineWidth', 1.5, 'Marker', marker_list{i},  'MarkerFaceColor', color_list{i},'MarkerSize',5);
        hold on
    end
    
    for i = 1 : 7
        xforfill = [x, fliplr(x)];
        yforfill = [error_lb(i, :), fliplr(error_ub(i, :))];
        fill(xforfill, yforfill, fill_color_list{i} , 'FaceAlpha',0.1,'EdgeAlpha', 0,'EdgeColor','r');
        hold on
    end
    title(title_list(plot_idx), 'Interpreter','latex');
    xlim([x(1) x(end)]);
    xlabel('average worker reliability','fontsize',26);
    ylabel('Prediction error', 'fontsize', 26)
    set(gca,'FontSize', 16)
%     [h,icons] = legend([A(1), A(2), A(3), A(4), A(5), A(6), A(7)],...
%         'MV','MV-D&S', 'OPT-D&S', 'PGD', 'Alg1', 'Alg2', 'Oracle', 'FontSize', 14);
    icons = findobj(icons,'Type','line');
    icons = findobj(icons,'Marker','none','-xor');
    set(icons,'MarkerSize',5);
end

%
% for plot_idx = 1 : 3
%     subplot(1, 3, plot_idx);
%     error_mean = [mean_mv(plot_idx, :); mean_mv_em(plot_idx, :); mean_pgd(plot_idx, :); mean_alg1(plot_idx, :); mean_alg2(plot_idx, :); mean_oracle(plot_idx, :)];
%     error_std = [std_mv(plot_idx, :); std_mv_em(plot_idx, :); std_pgd(plot_idx, :); std_alg1(plot_idx, :); std_alg2(plot_idx, :); std_oracle(plot_idx, :)];
%
%     error_lb = error_mean - error_std;
%     error_ub = error_mean + error_std;
%     for i = 1 : 6
%         A(i) = plot(x, error_mean(i, :), 'Color', color_list{i}, 'LineStyle', '-',...
%             'LineWidth', 1.5, 'Marker', marker_list{i},  'MarkerFaceColor', color_list{i},'MarkerSize',5);
%         hold on
%     end
%
%     for i = 1 : 6
%         xforfill = [x, fliplr(x)];
%         yforfill = [error_lb(i, :), fliplr(error_ub(i, :))];
%         fill(xforfill, yforfill, fill_color_list{i} , 'FaceAlpha',0.1,'EdgeAlpha', 0,'EdgeColor','r');
%         hold on
%     end
%     title(title_list(plot_idx), 'Interpreter','latex');
%     xlabel('Average worker reliability','fontsize',26);
%     ylabel('Prediction error', 'fontsize', 26)
%     xlim([x(1) x(end)]);
%     set(gca,'FontSize', 16)
%     [h,icons] = legend([A(1), A(2), A(3), A(4), A(5), A(6)],...
%         'MV','MV-D&S', 'PGD', 'Alg1', 'Alg2', 'Oracle', 'FontSize', 14);
%     icons = findobj(icons,'Type','line');
%     icons = findobj(icons,'Marker','none','-xor');
%     set(icons,'MarkerSize',5);
% end