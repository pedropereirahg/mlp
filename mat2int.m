function Y = mat2int(Y)
    for i = 1:size(Y,1)
        [~,indexMax] = max(Y(i,:));
        Y(i,:) = 0;
        Y(i, indexMax) = 1;
    end
end