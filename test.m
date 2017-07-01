function [Y,Z] = calc_Saida(X,A,B)
    Zin = X*A';
    N = size(X,1);
    Z = 1./(1+exp(-Zin)); %Z(N,nh)
    Yin = B*[ones(N,1),Z]';
    Y = (1./(1+exp(-Yin)))'; 
    
end
