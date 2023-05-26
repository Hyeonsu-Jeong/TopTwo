function [acc_pgd] = run_pgd(f, f_sep, C, g, h)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[N, M] = size(f);
K = max(f, [], 'all');
num_worker = N;
num_task = M;
num_class = K;
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
    score(:, i) = sum(f_sep(:, :, i) .* repmat(weights, 1, num_task), 1);
end
[estm_g, estm_h] = top2(score);

acc_pgd(1) = nnz(g == estm_g);
acc_pgd(2) = nnz(h == estm_h);
acc_pgd(3) = nnz((g == estm_g) & (h == estm_h));
end

