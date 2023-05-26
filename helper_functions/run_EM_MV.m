function [acc_em] = run_EM_MV(Z,g,h,valid_index,Nround)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n,k,m] = size(Z);
q = mean(Z,3);
q = q ./ repmat(sum(q,2),1,k);
mu = zeros(k,k,m);
mode=1;

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

[estm_g, estm_h] = top2(q);
acc_em(1) = nnz(g(valid_index) == estm_g(valid_index));
acc_em(2) = nnz(h(valid_index) == estm_h(valid_index));
acc_em(3) = nnz((g(valid_index) == estm_g(valid_index)) & (h(valid_index) == estm_h(valid_index)));
end

