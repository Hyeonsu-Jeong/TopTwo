function [acc_em] = run_EM_Spectral(Z,g,h,valid_index,Nround)
%===================== EM with spectral method ==============
% method of moment
mode = 0;
[n,k,m] = size(Z);
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
for iter = 1:Nround
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
end
[I,J] = sort(-q');
estm_g = J(1, :)';
estm_h = J(2, :)';
acc_em(1) = nnz(g(valid_index) == estm_g(valid_index));
acc_em(2) = nnz(h(valid_index) == estm_h(valid_index));
acc_em(3) = nnz((g(valid_index) == estm_g(valid_index)) & (h(valid_index) == estm_h(valid_index)));
end