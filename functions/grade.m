function [accuracy] = grade(g,h, hatg, hath)


    accuracy = zeros(3, 1);
    for i = 1:length(g)
        if g(i) == hatg(i)
            accuracy(1) = accuracy(1) + 1;
        end

        if h(i) == hath(i)
            accuracy(2) = accuracy(2) + 1;
        end

        if (g(i) == hatg(i)) && (h(i) == hath(i))
            accuracy(3) = accuracy(3) + 1;
        end
    end
end

