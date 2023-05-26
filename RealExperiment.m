clearvars; close all; clc;
addpath(genpath(pwd));


error_mv = zeros(1, 3);
error_em_mv = zeros(1, 3);
error_em_sm = zeros(1, 3);
error_pgd = zeros(1, 3);
error_multispa_kl = zeros(1, 3);
error_multispa_em = zeros(1, 3);
error_mmsr = zeros(1, 3);
error_toptwo1 = zeros(1, 3);
error_toptwo2 = zeros(1, 3);


color=0;

if color == 1
    [f, g, h, M, N, K] = LoadDataset_color();
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
else
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_Adult2();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_bluebird();
    [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_Dog();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_Flag();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_Food();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_Plot();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_RTE();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_trec();
%     [F,f,F_orig,f_orig,g,K,M,N,conf_mat] = LoadDataset_web();
    h = zeros(size(g));
end

valid_index = find(g>0);
N_valid=length(valid_index);
N_round=10;


%%% MV-EM
[A] = convert_for_comp(f_orig);
Z = zeros(N,K,M);
for i = 1:size(A,1)
    Z(A(i,1),A(i,3),A(i,2)) = 1;
end
error_em_mv = 1-run_EM_MV(Z,g,h,valid_index,N_round) / N;

%%% Spectral-EM
% [A] = convert_for_comp(f_orig);
% Z = zeros(N,K,M);
% for i = 1:size(A,1)
%     Z(A(i,1),A(i,3),A(i,2)) = 1;
% end
% error_em_sm = 1-run_EM_Spectral(Z,g,h,valid_index,N_round) / N;

%%% MV
error_mv = 1 - run_MV(f, g, h) / N;

%%% TopTwo1
[acc_1, estm_p, estm_q] = run_TopTwo1(f, g, h);
error_toptwo1 = 1-acc_1 / N;

%%% TopTwo2
error_toptwo2 = 1-run_TopTwo2(f, g, h, estm_p, estm_q)/N;

f_sep = zeros(M, N, K);
for i = 1 : K
    separate1 = zeros(M, N);
    separate1(find(f == i)) = 1;
    f_sep(:, :, i) = separate1;
end

pair_N = zeros(M);
for i = 1 : M
    for j = 1 : M
        if i == j
            pair_N(i, j) = 0;
        else
            pair_N(i, j) = sum(f(i, :) & f(j, :));
        end
    end
end

C = zeros(M);
for i = 1 : M
    for j = 1 : M
        if pair_N(i, j) ~= 0
            valid_idx = f(i, :) & f(j, :);
            C(i, j) = K/((K - 1) * pair_N(i,j)) * sum((f(i, :) ...
                == f(j, :)).* valid_idx) - 1/(K - 1); % multilabel equation in JMLR
        end
    end
end
%%% PGD
error_pgd = 1 - run_pgd(f, f_sep, C, g, h)/N;

%%% M-MSR
error_mmsr = 1-run_mmsr(f, f_sep, pair_N, C, g, h)/N;

%%% multiSPA-KL
[M_tens,M_mat,mean_vecs,params.M_tens_val,params.M_mat_val,N_valid_tens,N_valid_mat] = calc_annotator_moments(F,[2]);
[Gamma_est,pi_vec_est,list_g] = EstConfMat_SPA(M_mat,K);
[~,Gamma_est] = getPermutedMatrix(Gamma_est,list_g);

marg = combnk(1:M,2);           % marg defines the pairs of variables (or triples 2->3)
marg = num2cell(marg,2);        % convert it to cell
Y=get_second_order_stat(M_mat,marg);

opts = {}; opts.marg = marg; opts.max_iter = 10; opts.tol_impr = 1e-6;
Gamma_est=algorithm_init(Gamma_est,f_orig,N,K,M);
opts.A0 = Gamma_est;
opts.l0 = pi_vec_est;
I=K*ones(1,M);

[Gamma_KL,pi_vec_KL,Out] = N_CTF_AO_KL(Y,I,K,opts);

M_mod = length(list_g);
list_g1= list_g;
Gamma_hat_tensor = zeros(K,K,M_mod);
F_tensor = zeros(K,N,M_mod);

j=1;
for i=1:M_mod
    Gamma_hat_tensor(:,:,i) = Gamma_est{list_g1(i)};
    F_tensor(:,:,i) = F{list_g1(i)};
    j=j+1;
end

M_mod = M;
list_g2 = 1:M;
Gamma_hat_tensor2 = zeros(K,K,M);
F_tensor2 = zeros(K,N,M);
for i=1:M_mod
    Gamma_hat_tensor2(:,:,i) = Gamma_KL{list_g2(i)};
    F_tensor2(:,:,i) = F{list_g2(i)};
end

[estm_g, estm_h] = label_estimator(Gamma_hat_tensor2,F_tensor2,'ML');
acc_mspa_kl(1) = nnz(g(valid_index) == estm_g(valid_index));
acc_mspa_kl(2) = nnz(g(valid_index) == estm_g(valid_index));
acc_mspa_kl(3) = nnz(g(valid_index) == estm_g(valid_index));
error_multispa_kl = 1 - acc_mspa_kl / N;

Nround = 10;

%%% multiSPA-EM
[A] = convert_for_comp(f_orig);
Z = zeros(N,K,M);
for i = 1:size(A,1)
    Z(A(i,1),A(i,3),A(i,2)) = 1;
end
Gamma_hat_tensor_EM = zeros(K,K,M);
F_tensor_EM = zeros(K,N,M);
for i=1:M
    Gamma_hat_tensor_EM(:,:,i) = Gamma_est{i};
    F_tensor_EM(:,:,i) = F{i};
end

error_multispa_em = 1 - run_EM(Gamma_hat_tensor_EM,Z,g,h,valid_index,Nround)/N;

methods = ["MV"; "MV-D&S"; "OPT-D&S"; "PGD"; "MMSR"; "MultiSPA_KL"; "MultiSPA_EM"; "TopTwo1"; "TopTwo2"];
error = [error_mv; error_em_mv; error_em_sm; error_pgd; error_mmsr; error_multispa_kl; error_multispa_em; error_toptwo1; error_toptwo2];
if color == 1
    T = table(methods, error(:, 1), error(:, 2), error(:, 3));
    T.Properties.VariableNames = {'Method', 'error rate of g', 'error rate of h', 'error rate of pair'};
else
    T = table(methods, error(:, 1));
    T.Properties.VariableNames = {'Method', 'error rate of g'};
end

T