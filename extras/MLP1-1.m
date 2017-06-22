%*************************************************
% X - Matriz com os dados de treinamento
% d - matriz de saida
% h - numero de neuronios
%*************************************************

function [A,B]=MLP1(X,d,h)
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
    alfa = bissecao_mlp(X,d,A,B,dJdA,dJdB,N)
    A = A - alfa*dJdA;
    B = B - alfa*dJdB;
    Y = calc_saida(X,A,B,N);
    erro = Y - d;
    EQM = 1/N*sum(sum(erro.*erro));
    vet_EQM = [vet_EQM;EQM]
    
    %pause
end
plot(vet_EQM);
Y = calc_saida(X,A,B,N)
end()


function Y=calc_saida(X,A,B,N)
Zin = [ones(N,1),X]*A';
Z = 1./(1 + exp(-Zin));
Yin = [ones(N,1),Z]*B';
Y = Yin;
end