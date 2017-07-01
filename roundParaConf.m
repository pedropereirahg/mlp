function Y=roundParaConf(Y)
    Total = size(Y,1);
    for i=1: Total
        [~,maiorVal] = max(Y(i,:));
        Y(i,:) = 0;
        Y(i,maiorVal) = 1;
    end