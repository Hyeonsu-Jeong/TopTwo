function [observed_matrix, ground_truth, most_confusing_answer, M, N, K] = LoadDataset_color()

fid = fopen('./Datasets/color.txt');
A = textscan(fid,'%f%f%f','delimiter','\t');

fid = fopen('./Datasets/color_truth.txt');
B = textscan(fid,'%f%f','delimiter','\t');

fid = fopen('./Datasets/color_conf.txt');
conf = textscan(fid,'%f%f','delimiter','\t');


item_id = A{2}+1;
annotator_id=A{1}+1;
annotator_res=A{3}+1;
ground_truth = B{2}+1;
most_confusing_answer = conf{2}+1;

M = max(annotator_id);
N = max(item_id);
K = max(ground_truth);
observed_matrix = zeros(M,N); %annotator labels

C = [annotator_id item_id];

for i=1:length(C)
    observed_matrix(C(i,1),C(i,2))=annotator_res(i);
end
