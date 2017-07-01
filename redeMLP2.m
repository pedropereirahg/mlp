function [] = redeMLP2()
Xt = [0 0;0 1;1 0;1 1];
Yd = [0;1;1;0];
nh = 4;

ne = size(Xt, 2);
[N, ns] = size(Yd);
Xt = [ones(N, 1), Xt];

A = ones(nh, ne + 1)/5
B = ones(ns, nh + 1)/5

[Y, erro, EQM, ~, ~, ~] = calc_grad(Xt, Yd, A, B, N);
norm(EQM)
alfa = 0.9;

while(norm(EQM) >= 1e-5)
    
    [~, ~, EQM, ~, gradA, gradB] = calc_grad(Xt, Yd, A, B, N);
    B = B - alfa*gradB;
    A = A - alfa*gradA;
    
    disp(sprintf('Erro =%2.7f',norm(EQM)));
    
end

disp(Y);

end

function [Y, erro, EQM, g, gradA, gradB] = calc_grad(X, Yd, A, B, N)
Zin = X*A';
Z = 1./(1+exp(-Zin));
Yin=[ones(N,1),Z]*B';
Y = 1./(1+exp(-Yin));

erro = Y-Yd;
EQM = sum(sum(erro.*erro))/N;

gradB = 1/N*(erro.*(Y.*(1-Y)))'*[ones(N,1),Z];
DJDZ = (erro.*(Y.*(1-Y)))*B(:,2:end);
gradA = 1/N*(DJDZ.*(Z.*(1-Z)))'*X;
g = [gradA(:);gradB(:)];
end