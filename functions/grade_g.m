function [accuracy] = grade_g(g, hatg)

    accuracy = 0;
    for i = 1:length(g)
        if g(i) == hatg(i)
            accuracy = accuracy + 1;
        end
    end
    accuracy = 1 - accuracy / length(g);
end

