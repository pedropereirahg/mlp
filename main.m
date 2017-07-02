function main()

    path = 'images/*.txt';
    files = dir(path);

    X = cell(size(files));
    Yd = [];
    nh = 5;
    nkFold = 5;

    i = 1;

    for file = files'
        X{i} = load([file.folder '\' file.name])';

        if ~isempty(strfind(file.name, 'train_5a_')) %Letra X
            Yd = [Yd;[0,0,1]];
        end
        if ~isempty(strfind(file.name, 'train_53_')) %Letra S
            Yd = [Yd;[0,1,0]];
        end
        if ~isempty(strfind(file.name, 'train_58_')) %Letra Z
            Yd = [Yd;[1,0,0]];
        end

        i = i + 1;
    end

    X = cell2mat(X);

    i = 1;

    grupoPorFold = size(X,1)/nkFold;

    Indices = 1:size(X,1);

    Yacerto = cell(1, nkFold);

    while i <= nkFold
        c = randperm(size(Indices,2), grupoPorFold);  
        [trainX, trainYd, testX, testYd] = kfold(X, Yd, Indices(c));

        %SEM VALIDAR
        %[A,B] = redeNeural(trainX,trainYd,10);

        %VALIDANDO
        [A,B] = mlp(trainX, trainYd, nh, grupoPorFold);

        [Ntr,~] = size(testX);

        %TESTANDO
        [Yr,~] = feed_foward([ones(Ntr,1),testX], A, B);
        Yacerto{i} = acerto(Yr,testYd);

        %RESETRA PRO PROX FOLD
        Indices(c) = [];
        i = i+1;
    end

    letraYd = paraLetra(testYd);
    letraYr = paraLetra(roundParaConf(Yr));
    [C,order] = confusionmat(letraYd,letraYr);
    C

    %imagesc(C)
    %colorbar;
    %axis on;
    
    Yacerto = cell2mat(Yacerto);
    
    Yacerto
    mean(Yacerto)

end

function txAcerto = acerto(Yr,testYd)
    Yr = round(Yr);
    Total = size(Yr,1);
    cont =0;
    for i =1: Total
        mi = testYd(i,:);
        if mi == Yr(i,:);
            cont = cont+1;
        end
    end
    txAcerto = (cont*100)/Total;
end
