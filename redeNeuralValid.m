function [Amelhor,Bmelhor] = redeNeural(X,Yd,nh,numVal)
%function [Yr,txAcerto] = redeNeural(X,Yd,nh)

c=randperm(size(X,1),numVal); 
[X, Yd, valX,valYd] = grupos(X,Yd,c);
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
nepocasmax = 10;
veterro = [];
veterro = [veterro;EQM];
% veterro = [veterro;norm(g)];
%while norm(g)>1e-3 && nepocas < nepocasmax
[Ntr,~] = size(valX);
[Yval,~] = test([ones(Ntr,1),valX],A,B);
erro1 = (Yval - valYd);
erro1 = sum(sum(erro1.*erro1))/N;
Amelhor= A;
Bmelhor= B;

while  nepocas < nepocasmax
    %nepocas = nepocas +1;
    d = -g;
    alfa = bissecao_mlp(X,Yd,A,B,gA,gB,N);
    A = A - alfa*gA;
    B = B - alfa*gB;
    [Yr,~] = calc_Saida(X, A, B);
    erro = (Yr - Yd);
    EQM = sum(sum(erro.*erro))/N;
    veterro = [veterro;EQM];
    %disp(sprintf('Norm =%2.7f, nepocas=%d, alfa = %1.3f',norm(g),nepocas,alfa));
    [g,gA, gB] = calc_grad(X, Yd, A, B, N);
    %hold on;
    %plot(veterro);
    [Yval,~] = test([ones(Ntr,1),valX],A,B);
    erro2 = (Yval - valYd);
    erro2 = sum(sum(erro2.*erro2))/N;
    %erro1
    %erro2
    if(erro2< erro1)
        nepocas=0;
        erro1= erro2;
        Amelhor= A;
        Bmelhor= B;
        Ymelhor= Yval;
    else
        nepocas = nepocas +1;
    end
end

plot(veterro);
Yr = round(Yr);
cont = 0;
Total = size(Yr,1);
for i =1: Total
   mi = Yd(i,1);
   if mi == Yr(i,1)
       cont = cont+1;
   end
end

txAcerto = (cont*100)/Total;




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

function [TRX,TRY,TSTX,TSTY]=grupos(X,Yd,c)
    TSTX=X(c,:);  % output matrix
    TSTY=Yd(c,:);

    TRX = X;
    TRX(c,:)= [];
    TRY = Yd;
    TRY(c,:) = [];
end

