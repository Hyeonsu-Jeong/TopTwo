function [acc_mmsr] = run_TopTwo1(f, f_sep, pair_N, C, g, h)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
X = abs(C);
num_worker = size(C, 1);
num_tasks = size(f, 2);
num_class = max(f, [], 'all');
[m, n] = size(X);
idx = abs(sign(pair_N));
idx = (idx==1); % the positions of the observed entries
u = rand(m, 1);
v = rand(n,1);

sparsity = min(sum(sign(pair_N)));
F_parameter = floor(sparsity/2) - 1;
for t  = 1:10000
    v_pre = v;
    u_pre = u;
    for j = 1:n
        target_v = X(:,j);
        target_v = target_v(idx(:,j))./u(idx(:,j));
        a = mean(remove(target_v, F_parameter, v(j), num_tasks));
        if isnan(a)
            v(j) = v(j);
        else
            v(j) = a ;
        end
    end

    for i = 1:m
        target_u = X(i,:)';
        target_u = target_u(idx(i,:))./v(idx(i,:));
        a  = mean(remove(target_u, F_parameter, u(i), num_tasks));
        if isnan(a)
            u(i) = u(i);
        else
            u(i) = a;
        end

    end

    M=u*v';
    if norm(u * v' - u_pre * v_pre', 'fro') < 1e-10
        break
    end
end

k = sqrt(norm(u)/norm(v));
x_track_1 = u / k;
x_track_2 = sign_determination_valid(C, x_track_1);
x_track_3 = min(x_track_2, 1-1./sqrt(num_tasks));
x_MSR = max(x_track_3, -1/(num_class-1)+1./sqrt(num_tasks));
% prediction
probWorker = x_MSR.*(num_class-1)/(num_class)+1/num_class; % x is s in JMLR, which is not probability
weights = log(probWorker.*(num_class-1)./(1-probWorker));
score = zeros(num_class, num_tasks);
for i = 1 : num_class
    score(i, :) = sum(f_sep(:, :, i) .* repmat(weights, 1, num_tasks), 1);
end
[estm_g, estm_h] = top2(score');

acc_mmsr(1) = nnz(g == estm_g);
acc_mmsr(2) = nnz(h == estm_h);
acc_mmsr(3) = nnz((g == estm_g) & (h == estm_h));

end

