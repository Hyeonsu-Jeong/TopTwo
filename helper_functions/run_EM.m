function [acc_em] = run_EM(mu,Z,g,h,valid_index,Nround)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n,k,m] = size(Z);
% EM update
for iter = 1:Nround
    q = zeros(n,k);
    for i=1:m
        tmp = mu(:,:,i);
        tmp(find(tmp ==0)) = eps;
        tmp(find(isnan(tmp))) = eps;
        q = q + Z(:,:,i)*log(tmp);
        %q = q + Z(:,:,i)*log(mu(:,:,i));
    end
    q = exp(q);
    q = bsxfun(@rdivide,q,sum(q,2));

    for i = 1:m
        mu(:,:,i) = (Z(:,:,i))'*q;
        
        mu(:,:,i) = AggregateCFG(mu(:,:,i),0);
        tmp_vec = sum(mu(:,:,i));
        indx = find(tmp_vec > 0);
        mu(:,indx,i) = bsxfun(@rdivide,mu(:,indx,i),tmp_vec(indx));
    end
end
[I,J] = sort(-q');
estm_g = J(1, :)';
estm_h = J(2, :)';
acc_em(1) = nnz(g(valid_index) == estm_g(valid_index));
acc_em(2) = nnz(h(valid_index) == estm_h(valid_index));
acc_em(3) = nnz((g(valid_index) == estm_g(valid_index)) & (h(valid_index) == estm_h(valid_index)));
end