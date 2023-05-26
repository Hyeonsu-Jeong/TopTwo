function [F,f,F_orig,f_orig,y,K,M,N,conf_mat] = LoadDataset_bluebird()

%fid = fopen('/nfs/stak/users/ibrahish/Semester 3/Research/Code/CrowdSourcingAMT/Datasets/bluebird.txt');
fid = fopen('./Datasets/bluebird.txt');
A = textscan(fid,'%f%f%f','delimiter','\t');

fid = fopen('./Datasets/bluebird_truth.txt');
B = textscan(fid,'%f%f','delimiter','\t');

item_id = A{1};
annotator_id=A{2};
annotator_res=A{3};
ground_truth = B{2};

M = max(annotator_id);
N = max(item_id);
K = max(ground_truth);
f = zeros(M,N); %annotator labels

y = ground_truth;
C = [annotator_id item_id];

for i=1:length(C)
    f(C(i,1),C(i,2))=annotator_res(i);
end
f_orig=f;

conf_mat= ones(K,N,M);
% for i=1:N
%     g=f(:,i);
%     m=mode(g(g>0));
%     idx = g==0;
%     g(idx)=m;
%     f(:,i)=g;
%     conf_mat(m,i,idx)=0;
% end

J=sum(f>0,2);
[~,I]=sort(J,'descend');
% 
M=30;
f=f(I(1:M),:);
f_orig=f_orig(I(1:M),:);

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

end

