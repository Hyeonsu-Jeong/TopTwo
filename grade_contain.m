function [accuracy] = grade_contain(hatg, hath, ground_truth)

accuracy = 0;
for i = 1 : length(ground_truth)
    if hatg(i) == ground_truth(i) || hath(i) == ground_truth(i)
        accuracy = accuracy + 1;
    end
end
accuracy = 1 - accuracy / length(ground_truth);
end

