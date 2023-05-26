clearvars; close all; clc;
addpath(genpath(pwd));

% Hyperparameters
total_worker = 50;
total_task = 500;
total_label = 5;

p_min = 0.001;
p_max = 0.999;
q_min = 0.501;
q_max = 0.599;

sampling_list = 0.1:0.1:1.0;
rep_time = 30;

error_mv = zeros(size(sampling_list, 2), rep_time, 3);
error_toptwo1 = zeros(size(sampling_list, 2), rep_time, 3);
error_toptwo2 = zeros(size(sampling_list, 2), rep_time, 3);
error_oracle = zeros(size(sampling_list, 2), rep_time, 3);
error_em_mv = zeros(size(sampling_list, 2), rep_time, 3);
% error_em_sm = zeros(size(sampling_list, 2), rep_time, 3);
error_pgd = zeros(size(sampling_list, 2), rep_time, 3);
error_ebcc = zeros(size(sampling_list, 2), rep_time, 3);
error_mmsr = zeros(size(sampling_list, 2), rep_time, 3);
error_multispa = zeros(size(sampling_list, 2), rep_time, 3);
error_multispa_kl = zeros(size(sampling_list, 2), rep_time, 3);
error_multispa_em = zeros(size(sampling_list, 2), rep_time, 3);

for sampling_idx = 1 : size(sampling_list, 2)
    for times = 1:rep_time

        worker_reliability = (p_max - p_min) * rand(total_worker, 1) + p_min;
        task_difficulty = (q_max - q_min) * rand(total_task, 1) + q_min;

        [observed_matrix, ground_truth, most_confusing_answer] = Generate_TopTwo_Dataset(total_worker, total_task, total_label, worker_reliability, task_difficulty);
        
        % sampling confusion matrix
        disp([num2str(sampling_idx), ',', num2str(times)]);
        sampling_ratio = sampling_list(sampling_idx);
        sampling_matrix = zeros(total_worker, total_task);
        for i = 1 : total_worker
            for j = 1 : total_task
                if rand() < sampling_ratio
                    sampling_matrix(i, j) = observed_matrix(i, j);
                end
            end
        end

        [F,f,F_orig,f_orig,g,h,K,M,N,conf_mat,p,q] = LoadDataset(sampling_matrix, ground_truth, most_confusing_answer, worker_reliability, task_difficulty);

        valid_index = find(g>0);
        N_valid=length(valid_index);
        N_round=1;

        %%% MV-EM
        [A] = convert_for_comp(f_orig);
        Z = zeros(N,K,M);
        for i = 1:size(A,1)
            Z(A(i,1),A(i,3),A(i,2)) = 1;
        end
        error_em_mv(sampling_idx, times, :) = 1-run_EM_MV(Z,g,h,valid_index,N_round) / total_task;

        %%% Spectral-EM
        [A] = convert_for_comp(f_orig);
        Z = zeros(N,K,M);
        for i = 1:size(A,1)
            Z(A(i,1),A(i,3),A(i,2)) = 1;
        end
        error_em_sm = 1-run_EM_Spectral(Z,g,h,valid_index,N_round) / total_task;

        %%% MV
        error_mv(sampling_idx, times, :) = 1 - run_MV(f, g, h) / total_task;

        %%% TopTwo1
        [acc_1, estm_p, estm_q] = run_TopTwo1(f, g, h);
        error_toptwo1(sampling_idx, times, :) = 1-acc_1 / total_task;

        %%% TopTwo2
        error_toptwo2(sampling_idx, times, :) = 1-run_TopTwo2(f, g, h, estm_p, estm_q)/total_task;

        %%% Oracle
        error_oracle(sampling_idx, times, :) = 1-run_TopTwo2(f, g, h, p, q)/total_task;

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
        error_pgd(sampling_idx, times, :) = 1 - run_pgd(f, f_sep, C, g, h)/total_task;
        
        %%% EBCC
%         error_ebcc(sampling_idx, times, :) = 1 - run_ebcc(f, g, h);

        %%% M-MSR
        error_mmsr(sampling_idx, times, :) = 1-run_mmsr(f, f_sep, pair_N, C, g, h)/total_task;

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
        acc_mspa_kl(2) = nnz(h(valid_index) == estm_h(valid_index));
        acc_mspa_kl(3) = nnz((g(valid_index) == estm_g(valid_index)) & (h(valid_index) == estm_h(valid_index)));
        error_multispa_kl(sampling_idx, times, :) = 1 - acc_mspa_kl / total_task;
        
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
        
        error_multispa_em(sampling_idx, times, :) = 1 - run_EM(Gamma_hat_tensor_EM,Z,g,h,valid_index,Nround)/total_task;
    end   
end
