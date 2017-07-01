function [A,B] = redeNeural(X,Yd,nh)
%function [Yr,txAcerto] = redeNeural(X,Yd,nh)
%X = [0 0;0 1;1 0;1 1];
%Yd = [0;1;1;0];
%load('dados.mat');
%X = Xtr;
%Yd = Ytr;
% antes = datetime('now');
%z = load('spambase.data');
%X = z(:,1:end-4);
%Yd = z(:,end);

%nh = 6;

[N,ne] = size(X);
ns = size(Yd,2);
X = [ones(N,1),X]; % X(N,ne+1)

A = rands(nh,ne+1)/5;
B = rands(ns,nh+1)/5; 

[Y,~] = calc_Saida(X,A,B);
erro = Y - Yd;

EQM = sum(sum(erro.*erro))/N;
[g,gA,gB] = calc_grad(X,Yd,A,B,N);
nepocas =0;
nepocasmax = 10000;
veterro = [];
veterro = [veterro;EQM];
% veterro = [veterro;norm(g)];
while norm(g)>1e-3 && nepocas < nepocasmax
 nepocas = nepocas +1;
    d = -g;
%     alfa = 0.9;
    alfa = bissecao_mlp(X,Yd,A,B,gA,gB,N);
    A = A - alfa*gA;
    B = B - alfa*gB;
    [Yr,~] = calc_Saida(X, A, B);
    erro = (Yr - Yd);
    EQM = sum(sum(erro.*erro))/N;
%   veterro = [veterro;norm(g)];
    veterro = [veterro;EQM];
   %disp(sprintf('Norm =%2.7f, nepocas=%d, alfa = %1.3f',norm(g),nepocas,alfa));
   [g,gA, gB] = calc_grad(X, Yd, A, B, N);
   %hold on;
   %plot(veterro);
end

% plot(veterro);
% Yr = round(Yr);
% cont = 0;
% Total = size(Yr,1);
% for i =1: Total
%    mi = Yd(i,1);
%    if mi == Yr(i,1)
%        cont = cont+1;
%    end
% end
% 
% txAcerto = (cont*100)/Total;


end


function [Y,Z] = calc_Saida(X,A,B)
    Zin = X*A';
    N = size(X,1);
    Z = 1./(1+exp(-Zin)); %Z(N,nh)
    Yin = B*[ones(N,1),Z]';
    Y = (1./(1+exp(-Yin)))';
    
end

function [g,gradA, gradB]=calc_grad(X,Yd,A,B,N)
[Y,Z] = calc_Saida(X,A,B);

erro = Y-Yd;
gradB = 1/N*(erro.*(Y.*(1-Y)))'*[ones(N,1),Z];
DJDZ = (erro.*(Y.*(1-Y)))*B(:,2:end);
gradA = 1/N*(DJDZ.*(Z.*(1-Z)))'*X;
g = [gradA(:);gradB(:)];
end


