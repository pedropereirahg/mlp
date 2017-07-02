function [Amelhor,Bmelhor] = mlp(X, Yd, nh, numVal)
    
    c = randperm(size(X,1), numVal); 
    [X, Yd, valX,valYd] = kfold(X, Yd, c);

    [N,ne] = size(X);
    ns = size(Yd,2);
    X = [ones(N,1),X];

    A = rands(nh,ne+1)/5;
    B = rands(ns,nh+1)/5; 

    [Y,~] = feed_foward(X, A, B);
    erro = Y - Yd;

    EQM = sum(sum(erro.*erro))/N;
    [gA, gB] = gradient(X,Yd,A,B,N);
    nepocas =0;
    nepocasmax = 10;
    veterro = [];
    veterro = [veterro;EQM];
    
    [Ntr,~] = size(valX);
    [Yval,~] = feed_foward([ones(Ntr,1),valX],A,B);
    erro1 = (Yval - valYd);
    erro1 = sum(sum(erro1.*erro1))/N;
    Amelhor = A;
    Bmelhor = B;

    while  nepocas < nepocasmax
        
        alfa = bissecao_mlp(X,Yd,A,B,gA,gB,N);
        A = A - alfa*gA;
        B = B - alfa*gB;
        [Yr,~] = feed_foward(X, A, B);
        erro = (Yr - Yd);
        EQM = sum(sum(erro.*erro))/N;
        veterro = [veterro;EQM];
        
        [gA, gB] = gradient(X, Yd, A, B, N);
        
        [Yval,~] = feed_foward([ones(Ntr,1),valX],A,B);
        erro2 = (Yval - valYd);
        erro2 = sum(sum(erro2.*erro2))/N;
        
        if (erro2 < erro1)
            nepocas = 0;
            erro1 = erro2;
            Amelhor = A;
            Bmelhor = B;
            Ymelhor = Yval;
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
