function [Y,Z] = feed_foward(X, A, B)
    N = size(X,1);    
    Zin = X*A';
    Z = 1./(1+exp(-Zin));
    Yin = B*[ones(N,1),Z]';
    Y = (1./(1+exp(-Yin)))';
end