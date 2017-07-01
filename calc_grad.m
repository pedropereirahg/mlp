function [dJdA,dJdB]=calc_grad(X,d,A,B,N)


% Zin = [ones(N,1),X]*A;
% Z = 1./(1 + exp(-Zin));
% k = [ones(N,1),Z];
% Yin = [ones(N,1),Z]*B';
% Y = Yin;
% erro = Y - d;
%Zin = X*A';
% N = size(X,1);
Zin = X*A';
Z = 1./(1+exp(-Zin)); %Z(N,nh)
Yin = B*[ones(N,1),Z]';
%Yin = [ones(N,1),Z]*B';
%Y = Yin;
Y = (1./(1+exp(-Yin)))';
erro = Y - d;

% dJdB = 1/N*(erro'*[ones(N,1),Z]);
% dJdZ = erro*B;
% dJdZ = dJdZ(:,2:end);
% dJdA = 1/N * (dJdZ.*((1-Z).*Z))'*[ones(N,1),X];
% 

dJdB = 1/N*(erro.*(Y.*(1-Y)))'*[ones(N,1),Z];
dJdZ = (erro.*(Y.*(1-Y)))*B(:,2:end);
dJdA = 1/N*(dJdZ.*(Z.*(1-Z)))'*X;
end


% function [Y,Z] = calc_Saida(X,A,B)
%     Zin = X*A';
%     N = size(X,1);
%     Z = 1./(1+exp(-Zin)); %Z(N,nh)
%     Yin = B*[ones(N,1),Z]';
%     Y = (1./(1+exp(-Yin)))';
%     
% end
% 
% function [g,gradA, gradB]=calc_grad(X,Yd,A,B,N)
% [Y,Z] = calc_Saida(X,A,B);
% 
% erro = Y-Yd;
% gradB = 1/N*(erro.*(Y.*(1-Y)))'*[ones(N,1),Z];
% DJDZ = (erro.*(Y.*(1-Y)))*B(:,2:end);
% gradA = 1/N*(DJDZ.*(Z.*(1-Z)))'*X;
% g = [gradA(:);gradB(:)];
% end
