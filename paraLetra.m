function letra=paraLetra(Y)
    Total = size(Y,1);
    letra = [];
    for i=1: Total
        
        if Y(i,:) == [0,0,1]
           letra = [letra;'X'];
        
        elseif Y(i,:) == [0,1,0]
           letra = [letra;'S'];
        
        elseif Y(i,:) == [1,0,0]
           letra = [letra;'Z'];     
        end
    end