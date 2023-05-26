clearvars; close all; clc;
addpath("Synt_Data\")

data_list = ["easy.mat", "hard.mat", "few_smart.mat", "high_variance.mat"];
color_list = {[0.4940 0.1840 0.5560], [0.8500 0.3250 0.0980], [0.9640 0.4270 0.6070], [0.3010 0.7450 0.9330], [0.4660 0.6740 0.1880], [0.7660 0.7740 0.1120], [0 1 0], [0.3 0.7 0], [0 0 1], [1 0 0], [0 0 0]};

marker_list = {'p', 'o',  '<', 'x','s','^', 'd', 'p', 'o', 'x', '^'};
fill_color_list = color_list;

xlabel_list = {'Easy', 'Hard', 'Few Smart', 'High Variance'};
ylabel_list = {'P($g \neq \hat{g}$)', 'P($h \neq \hat{h}$)', 'P($(g, h) \neq (\hat{g}, \hat{h}$))'};

x =  1:10;

fig = figure(1);
set(fig, 'OuterPosition', [100, 200, 1400, 1100])

plot_list = [3, 1, 2];
tiledlayout(3, 4, 'TileSpacing', 'Compact', 'Padding', 'Compact');
ylabel_list = ["$P((g, h) \neq P(\hat{g}, \hat{h}))$", "$P(g \neq \hat{g})$", "$P(h \neq \hat{h})$"];


for row= 1 : 3
    for col=1:4
        nexttile;
        data = data_list(col);
        
        load(data);

        error = cat(4, error_mv, error_em_mv, error_em_sm, error_pgd, error_mmsr, error_multispa_kl, error_multispa_em, error_ebcc, error_toptwo1, error_toptwo2, error_oracle);

        error_mean = squeeze(mean(error, 2));
        error_std = squeeze(std(error, 0, 2));
        p = plot_list(row);
        mu = squeeze(error_mean(:, p, :));
        sigma = squeeze(error_std(:, p, :));
        lb = mu - sigma;
        ub = mu + sigma;
        A = gobjects(1, 11);

        for i = 1 : 11
            A(i) = plot(x, mu(:, i), 'Color', color_list{i}, 'LineStyle', '-',...
                'LineWidth', 1.5, 'Marker', marker_list{i},  'MarkerFaceColor', color_list{i},'MarkerSize',5);
            hold on
        end

        for i = 1 : 11
            xforfill = [x, fliplr(x)];
            yforfill = [lb(:, i)', fliplr(ub(:, i)')];
            fill(xforfill, yforfill, fill_color_list{i} , 'FaceAlpha',0.1,'EdgeAlpha', 0,'EdgeColor','r');
            hold on
        end

        if row == 1 && col == 1
            [h,icons] = legend([A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10), A(11)],...
    'MV','MV-D&S', 'OPT-D&S', 'PGD', 'M-MSR', 'MultiSPA-KL', 'MultiSPA-EM', 'EBCC', 'TopTwo1', 'TopTwo2', 'Oracle','FontSize', 14, 'Orientation', 'Horizontal');
            icons = findobj(icons,'Type','line');
            icons = findobj(icons,'Marker','none','-xor');
            set(icons,'MarkerSize',5);
            h.Layout.Tile = 'north';
        end

        if row == 1
            title(xlabel_list(col), 'fontweight','bold', 'fontsize', 20);
        end
        if row == 3
            xlabel('Avg, # of queries per task','fontsize',14);
        end
%         
        if col == 1
            ylabel(ylabel_list(row), 'fontsize', 16, 'Interpreter', 'latex')
        end
        set(gca,'LineWidth',1.5)
        xlim([x(2) x(end)])
    end
end
