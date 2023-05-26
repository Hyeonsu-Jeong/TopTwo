function [g,h] = top2(score)
[a, ~] = size(score);

g = zeros(a, 1);
h = zeros(a, 1);

for i = 1 : a
    [~, order] = sort(-score(i, :));
    g(i) = order(1);
    h(i) = order(2);
end

