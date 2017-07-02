function [dJdA,dJdB] = gradient(X, Yd, A, B, N)

    [Y,Z] = feed_foward(X, A, B);

    erro = Y - Yd;
    dJdB = 1/N*(erro.*(Y.*(1-Y)))'*[ones(N,1),Z];
    dJdZ = (erro.*(Y.*(1-Y)))*B(:,2:end);
    dJdA = 1/N*(dJdZ.*(Z.*(1-Z)))'*X;
end