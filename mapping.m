function ret = mapping(Y, map)
    ret = cell(size(Y,1), 1);
    for i = 1:size(Y,1)
        
        j = 1;
        while j <= size(map,1)
            if isequal(Y(i,:), map{j}{2})
                ret{i} = map{j}{1};
                break
            end
            j = j + 1;
        end
    end
end