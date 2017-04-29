%*************************************************
% X - Matriz com os dados de treinamento
% d - matriz de saida
% h - numero de neuronios
%*************************************************

function [A,B]=MLP(X,d,h)
[N,ne] = size(X);
ns = size(d,2);

% Inicializa os pesos
A = rands(h,ne+1);
B = rands(ns,h+1);
% Calcula a saida da RN
Y = calc_saida(X,A,B,N);
erro = Y - d;
EQM = 1/N*sum(sum(erro.*erro));
nepmax = 10000;
nepocas =0;
alfa = 1;
vet_EQM = [];
vet_EQM = [vet_EQM;EQM]

while EQM > 1.0e-5 & nepocas < nepmax
    nepocas = nepocas +1;
    [dJdA,dJdB]=calc_grad(X,d,A,B,N);
    Aaux = A - alfa*dJdA;
    Baux = B - alfa*dJdB;
    Y = calc_saida(X,Aaux,Baux,N);
    erro = Y - d;
    EQMaux = 1/N*sum(sum(erro.*erro));
    while EQMaux>EQM
        alfa = alfa*0.9;
        Aaux = A - alfa*dJdA;
        Baux = B - alfa*dJdB;
        Y = calc_saida(X,Aaux,Baux,N);
        erro = Y - d;
        EQMaux = 1/N*sum(sum(erro.*erro));
    end
    alfa = alfa/0.9;
    A = Aaux;
    B = Baux;
    EQM = EQMaux;    
    vet_EQM = [vet_EQM;EQM];
end
plot(vet_EQM);
Y = calc_saida(X,A,B,N)
end

function [dJdA,dJdB]=calc_grad(X,d,A,B,N)
Zin = [ones(N,1),X]*A';
Z = 1./(1 + exp(-Zin));
Yin = [ones(N,1),Z]*B';
Y = Yin;
erro = Y - d;

dJdB = 1/N*(erro'*[ones(N,1),Z]);
dJdZ = erro*B;
dJdZ = dJdZ(:,2:end);
dJdA = 1/N * (dJdZ.*((1-Z).*Z))'*[ones(N,1),X];
end


function Y=calc_saida(X,A,B,N)
Zin = [ones(N,1),X]*A';
Z = 1./(1 + exp(-Zin));
Yin = [ones(N,1),Z]*B';
Y = Yin;
end