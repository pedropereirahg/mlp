function [Amelhor,Bmelhor,TrnMed,ValMed] = mlp(X, Yd, nh, numVal, A, B)
    
    c = randperm(size(X,1), numVal); 
    [X, Yd, valX,valYd] = kfold(X, Yd, c);

    [N,ne] = size(X);
    ns = size(Yd,2);
    X = [ones(N,1),X];
    
    if ~exist('A', 'var')
        A = rand(nh,ne+1)/5;
    end
    
    if ~exist('B', 'var')
        B = rand(ns,nh+1)/5;
    end

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
    veterroval = [];
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
        veterroval = [veterroval;erro2];
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
    TrnMed = mean(veterro);
    ValMed = mean(veterroval);
    
    %plot(veterro);
    %saveas(gcf,'erro_train.png')
    %plot(veterroval);
    %saveas(gcf,'erro_val.png')
end

