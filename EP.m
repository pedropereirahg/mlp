function EP()

X =[];
Yd=[];

files = dir('build/hog/ppc_32_cpb_1_o_9/images/*.txt');
%files = dir('images/*.txt');

for file = files'
    X = [X;load(file.name)'];
    %file.name
    if ~isempty(findstr(file.name,'train_5a_')) %Letra X
        Yd = [Yd;[0,0,1]];
    end
    if ~isempty(findstr(file.name,'train_53_')) %Letra S
        Yd = [Yd;[0,1,0]];
    end
    if ~isempty(findstr(file.name,'train_58_')) %Letra Z
        Yd = [Yd;[1,0,0]];
    end
end

i = 0;
k = 5;

%Testar K-fold
%X = [1,0,0,0,0;2,0,0,0,0;3,0,0,0,0; 4,0,0,0,0; 5,0,0,0,0];
%Yd = [1,0,0;2,0,0;3,0,0;4,0,0;5,0,0];

grupoPorFold = size(X,1)/k;

Indices = [1:size(X,1)];

Yacerto = [];

while i < k
    c=randperm(size(Indices,2),grupoPorFold);  
    [trainX,trainYd,testX,testYd] = grupos(X,Yd,Indices(c));
    
    %SEM VALIDAR
    %[A,B] = redeNeural(trainX,trainYd,10); #
    
    %VALIDANDO
    [A,B] = redeNeuralValid(trainX,trainYd,10,grupoPorFold);
    
    [Ntr,~] = size(testX);
    
    %TESTANDO
    [Yr,~] = test([ones(Ntr,1),testX],A,B);
    Yacerto = [Yacerto,acerto(Yr,testYd)];
    
    %RESETRA PRO PROX FOLD
    Indices(c) = [];
    i = i+1;
end

letraYd = paraLetra(testYd);
letraYr = paraLetra(roundParaConf(Yr));
[C,order] = confusionmat(letraYd,letraYr);

%imshow(C, [], 'InitialMagnification', 10000);
%imagesc(C, [], 'InitialMagnification', 10000);
imagesc(C)
colorbar;
axis on;

mean(Yacerto)

end

function txAcerto=acerto(Yr,testYd)
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

function [TRX,TRY,TSTX,TSTY]=grupos(X,Yd,c)
    TSTX=X(c,:); 
    TSTY=Yd(c,:);

    TRX = X;
    TRX(c,:)= [];
    TRY = Yd;
    TRY(c,:) = [];
end
