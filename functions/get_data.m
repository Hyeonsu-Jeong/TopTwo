function [observed_matrix, ground_truth] = get_data(name)

path = "dataset/"+name;
if exist(path, 'dir') ~= 7
    fprintf("Invalid dataset name\n");
end

crowd_data_file = load(path+"/crowd_data.txt");
ground_truth_file = load(path+"/ground_truth.txt");

A = max(crowd_data_file);
n = A(1);
m = A(2);
k = A(3);

observed_matrix = zeros(n, m);
ground_truth= zeros(m, 1);

for i = 1 : size(crowd_data_file, 1)
    observed_matrix(crowd_data_file(i, 1) + 1, crowd_data_file(i, 2) + 1) = crowd_data_file(i, 3) + 1;
end

for i = 1 : size(ground_truth_file, 1)
    ground_truth(ground_truth_file(i, 1) + 1) = ground_truth_file(i, 2) + 1;
end

end
