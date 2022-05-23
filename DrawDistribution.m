datasets = ["Adult2", "Dog", "Web", "Plot", "Flag", "Food", "Color"];

for idx = 1:7
    dataset = datasets(idx);
    observed_matrix = load(dataset+".txt");
    [N M] = size(observed_matrix);
    K = max(observed_matrix, [], 'all');
    label_distribution = zeros(M, K);
    for j = 1 : M
        for i = 1 : N
            if observed_matrix(i, j) ~= 0
                label_distribution(j, observed_matrix(i, j)) = label_distribution(j, observed_matrix(i, j)) + 1;
            end
        end
        label_distribution(j, :) = label_distribution(j, :) / sum(label_distribution(j, :));
        label_distribution(j, :) = sort(label_distribution(j, :), 'descend');
    end
    fig = figure;
    set(fig, 'OuterPosition', [270, 270, 670, 670]);
    x = 1 : K;
    m = mean(label_distribution, 1);
    error = std(label_distribution, 1);
    
    ub = min(1-m, error);
    lb = min(m, error);
    
    errorbar(x, m, lb, ub, 'LineWidth', 3);
    
    xticks(1:K);
    yticks(0:0.2:1);
    xlim([1 K]);
    ylim([0 1]);
    set(gca, 'FontSize', 20);
    set(gca, 'LineWidth', 2);
    title(dataset, 'Fontweight', 'bold', 'Fontsize', 30);
    xlabel('Label', 'Fontsize', 25);
    ylabel('Empirical probability', 'Fontsize', 25); 
    
    saveas(fig, "distribution/"+dataset+"_distribution.png");
end