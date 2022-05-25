clc
clear
close

dataset = "movie";

[Y_obs, ground_truth] = get_data(dataset);

trimmed_ground_truth = ground_truth;
trimmed_ground_truth(~any(Y_obs, 1)) = [];

Y_obs(:, ~any(Y_obs, 1)) = [];

[num_worker, num_task] = size(Y_obs);
num_class = max(Y_obs, [], 'all');

Y_obs_separate = zeros(num_worker, num_task, num_class);
for i = 1 : num_class
    separate1 = zeros(num_worker, num_task);
    separate1(find(Y_obs == i)) = 1;
    Y_obs_separate(:, :, i) = separate1;
end

N = zeros(num_worker);
for i = 1 : num_worker
    for j = 1 : num_worker
        if i == j
            N(i, j) = 0;
        else
            N(i, j) = sum(Y_obs(i, :) & Y_obs(j, :));
        end
    end
end

C = zeros(num_worker);
for i = 1 : num_worker
    for j = 1 : num_worker
        if N(i, j) ~= 0
            valid_idx = Y_obs(i, :) & Y_obs(j, :);
            C(i, j) = num_class/((num_class - 1) * N(i,j)) * sum((Y_obs(i, :) ...
                == Y_obs(j, :)).* valid_idx) - 1/(num_class - 1); % multilabel equation in JMLR
        end
    end
end

y = trimmed_ground_truth;
A = [];
Z = zeros(num_task, num_class, num_worker);
for i = 1 : num_class
    index = find(Y_obs_separate(:,:,i) ~= 0);
    [worker_idx , task_idx] = ind2sub([num_worker, num_task], index);
    class_idx = i * ones(size(task_idx));
    A = [A; task_idx, worker_idx, class_idx];
    index_Z = sub2ind(size(Z), task_idx, class_idx, worker_idx);
    Z(index_Z) = 1;
end

valid_index = find(y > 0);
%
n = num_task;
m = num_worker;
k = num_class;

Nround = 1;
mode = 1;

error1_predict = zeros(1, Nround);
error2_predict = zeros(1, Nround);

t = zeros(n,k-1);
for l = 1:k-1
    U = zeros(n,m);
    for i = 1:size(A,1)
        U(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    
    B = U - ones(n,1)*(ones(1,n)*U)/n;
    [U S V] = svd(B);
    u = U(:,1);
    v = V(:,1);
    u = u / norm(u);
    v = v / norm(v);
    pos_index = find(v>=0);
    if sum(v(pos_index).^2) >= 1/2
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_KOS = mean(y(valid_index) ~= (J(valid_index)));
%===================== Ghosh-SVD ================
t = zeros(n,k-1);
for l = 1:k-1
    O = zeros(n,m);
    for i = 1:size(A,1)
        O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    
    [U S V] = svd(O);
    u = sign(U(:,1));
    if u'*sum(O,2) >= 0
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_GhostSVD = mean(y(valid_index) ~= (J(valid_index)));

%         % %===================== Ratio of Eigenvalues ================
t = zeros(n,k-1);
for l = 1:k-1
    O = zeros(n,m);
    for i = 1:size(A,1)
        O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    G = abs(O);
    %
    %             % ========== algorithm 1 =============
    %         [U S V] = svd(O'*O);
    %         v1 = U(:,1);
    %         [U S V] = svd(G'*G);
    %         v2 = U(:,1);
    %         v1 = v1./v2;
    %         u = O*v1;
    % ========== algorithm 2 =============
    R1 = (O'*O)./(G'*G+10^-8);
    R2 = (G'*G > 0)+1-1;
    [U S V] = svd(R1);
    v1 = U(:,1);
    [U S V] = svd(R2);
    v2 = U(:,1);
    v1 = v1./v2;
    u = O*v1;
    
    if u'*sum(O,2) >= 0
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_RatioEigen = mean(y(valid_index) ~= (J(valid_index)));

%===================== EM with majority vote ================
q = mean(Z,3);
q = q ./ repmat(sum(q,2),1,k);
mu = zeros(k,k,m);

% EM update
for i = 1:m
    mu(:,:,i) = (Z(:,:,i))'*q;
    mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
    
    for c = 1:k
        mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
    end
end

q = zeros(n,k);
for j = 1:n
    for c = 1:k
        for i = 1:m
            if Z(j,:,i)*mu(:,c,i) > 0
                q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
            end
        end
    end
    q(j,:) = exp(q(j,:));
    q(j,:) = q(j,:) / sum(q(j,:));
end

[g, h] = top2(q);
error_em_mv = [grade_g(g, ground_truth), grade_contain(g, h, ground_truth)];
%===================== EM with spectral method ==============
% method of moment
group = mod(1:m,3)+1;
Zg = zeros(n,k,3);
cfg = zeros(k,k,3);
for i = 1:3
    I = find(group == i);
    Zg(:,:,i) = sum(Z(:,:,I),3);
end

x1 = Zg(:,:,1)';
x2 = Zg(:,:,2)';
x3 = Zg(:,:,3)';

muWg = zeros(k,k+1,3);
muWg(:,:,1) = SolveCFG(x2,x3,x1);
muWg(:,:,2) = SolveCFG(x3,x1,x2);
muWg(:,:,3) = SolveCFG(x1,x2,x3);

mu = zeros(k,k,m);
for i = 1:m
    x = Z(:,:,i)';
    x_alt = sum(Zg,3)' - Zg(:,:,group(i))';
    muW_alt = (sum(muWg,3) - muWg(:,:,group(i)));
    mu(:,:,i) = (x*x_alt'/n) / (diag(muW_alt(:,k+1)/2)*muW_alt(:,1:k)');
    
    mu(:,:,i) = max( mu(:,:,i), 10^-6 );
    mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
    for j = 1:k
        mu(:,j,i) = mu(:,j,i) / sum(mu(:,j,i));
    end
end

% EM update

q = zeros(n,k);
for j = 1:n
    for c = 1:k
        for i = 1:m
            if Z(j,:,i)*mu(:,c,i) > 0
                q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
            end
        end
    end
    q(j,:) = exp(q(j,:));
    q(j,:) = q(j,:) / sum(q(j,:));
end

for i = 1:m
    mu(:,:,i) = (Z(:,:,i))'*q;
    
    mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
    for c = 1:k
        mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
    end
end

[g, h] = top2(q);
error_em_sm = [grade_g(g, ground_truth), grade_contain(g, h, ground_truth)];
%====================== PGD ==========================================
x_p = 0.5*ones(num_worker,1);
alpha=1e-5;
x1 = zeros(num_worker,1);
t = 0;
while sum(abs(x_p-x1))>1e-10
    x1=x_p;
    x_p = x_p + alpha*grad1(x_p,abs(C),N);
    x_p=min(x_p,1-1./sqrt(num_task));
    x_p=max(x_p,-1/(num_class-1)+1./sqrt(num_task));
    sum(abs(x_p-x1));
    t = t + 1;
    if t == 300000
        break
    end
end
probWorker = x_p.*(num_class-1)/(num_class)+1/num_class; % x is s in JMLR, which is not probability
predlabel = zeros(num_task,1);
Error = 0;
weights = log(probWorker.*(num_class-1)./(1-probWorker));
score = zeros(num_task, num_class);
for i = 1 : num_class
    score(:, i) = sum(Y_obs_separate(:, :, i) .* repmat(weights, 1, num_task), 1);
end
[g, h] = top2(score);
error_pgd = [grade_g(g, ground_truth), grade_contain(g, h, ground_truth)];
% ================================ TopTwo1 ================================
observed_matrix = Y_obs;

[N, M] = size(observed_matrix);
K = max(observed_matrix, [], 'all');
S = nnz(observed_matrix) / numel(observed_matrix);

shifted_matrix= zeros(K, N, M);
us = zeros(K-1, N);

for k = 1 : K-1
    observed_matrix_bin = observed_matrix;
    observed_matrix_bin((1 <= observed_matrix) & (observed_matrix <= k)) = -1;
    observed_matrix_bin(observed_matrix>k) = 1;
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

[g, h] = top2(-del_w);
error_alg1 = [grade_g(g, ground_truth), grade_contain(g, h, ground_truth)];
% ================================ TopTwo2 ================================


p = sum(observed_matrix == repmat(g', N, 1) | observed_matrix == repmat(h', N, 1), 2);
worker_degree = sum(observed_matrix > 0, 2);

p = p ./ worker_degree;
p = p - 2/K;
p = p / (1-2/K);

p = max(min(p, 0.999), 0.001);
norm_p = norm(p);

q = ones(M, 1) / K;
for j = 1 : M
    q(j) = q(j) - (del_w(j, g(j)) / norm_p);
end

q = max(min(q, 0.999), 0.501);
% q = 0.498 * (q - min(q))/(max(q)-min(q)) + 0.501;

g = zeros(M, 1);
h = zeros(M, 1);

for j = 1 : M
    first_coeff = log(K * q(j) * (p./(1-p)) + 1);
    second_coeff = log(K *(1-q(j)) * (p./(1-p)) + 1);
    
    sum_g = zeros(K, 1);
    sum_h = zeros(K, 1);
    
    for i = 1 : N
        if observed_matrix(i, j) ~= 0
            sum_g(observed_matrix(i, j)) = sum_g(observed_matrix(i, j)) + first_coeff(i);
            sum_h(observed_matrix(i, j)) = sum_h(observed_matrix(i, j)) + second_coeff(i);
        end
    end
    
    max_val = -1;
    for a = 1 : K
        for b = 1 : K
            if a ~= b && max_val < sum_g(a) + sum_h(b)
                max_val =  sum_g(a) + sum_h(b);
                g(j) = a;
                h(j) = b;
            end
        end
    end
end
error_alg2 = [grade_g(g, ground_truth), grade_contain(g, h, ground_truth)];
% ================================ MV =====================================

g = zeros(M, 1);
h = zeros(M, 1);
for j = 1 : M
    count = zeros(K, 1);
    for i = 1 : N
        if observed_matrix(i, j) ~= 0
            count(observed_matrix(i, j)) = count(observed_matrix(i, j)) + 1;
        end
    end
    [~, order] = sort(count, 'descend');
    g(j) = order(1);
    h(j) = order(2);
end
error_mv = [grade_g(g, ground_truth), grade_contain(g, h, ground_truth)];
% =========================================================================

fprintf("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\ \n", dataset, error_mv(2), error_alg1(2), error_alg2(2), error_em_mv(2), error_em_sm(2), error_pgd(2));
