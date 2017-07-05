function [ret, order] = confusionmat(v1, v2)
    order = union(unique(v1), unique(v2));
    ret = zeros(size(order, 1));
    for i = 1:size(v1)
       i1 = find(strcmp(order, v1{i}));
       i2 = find(strcmp(order, v2(i)));
       ret(i1, i2) = ret(i1, i2) + 1;
    end
end