function main(input, output, verbose)
    
    % Prevent calls without args        
    if ~exist('input', 'var') || ~exist('output', 'var')
        return
    end
    
    % Log for analysis performance
    if exist('verbose', 'var') && verbose == true
        log();
    end
    
    [path, data, expecOutput, nh, nkFold] = readInput(input);
    
    % First, try run aplication with loaded data
    % The weights (A,B) will be loaded
    try
        newPath = path;
        newExpecOutput = expecOutput;
        
        load(data);
        
        if ~strcmp(newPath, path) || ~strcmp(newExpecOutput, expecOutput)
            error('Input file has changed.');
        end
    catch
        % Construct X and expectedOutput (Yd)
        [X, Yd] = expectedOutput(expecOutput, path);
    end
    
    % Run the principal loop
    [Y, A, B, C, order,vetErTrain,vetErVal,vetErTst] = runMLP(X, Yd, nh, nkFold);
    
    saveOutput(output, Y, C, order, vetErTrain, vetErVal, vetErTst);
    
    if exist('verbose', 'var') && verbose == true
        log(true);
    end
    clear verbose input output nh nkFold newPath newExpecOutput;
    
    save(data);
end

function [Y, A, B, C, order,vetErTrain,vetErVal,vetErTst] = runMLP(X, Yd, nh, nkFold, ALoad, BLoad)
    
    Y = cell(1, nkFold);
    n = size(X,1);
    Indices = 1:n;
    grupoPorFold = n/nkFold;
    
    vetErTst = [];
    vetErTrain = [];
    vetErVal = [];
    
    i = 1;
    while i <= nkFold
        c = randperm(size(Indices,2), grupoPorFold);  
        [trainX, trainYd, testX, testYd] = kfold(X, Yd, Indices(c));
        
        % Training and validation group
        if exist('ALoad', 'var') && exist('BLoad', 'var')
            [A,B,erroTest,erroVal] = mlp(trainX, trainYd, nh, grupoPorFold, ALoad, BLoad);
        else
            [A,B,erroTest,erroVal] = mlp(trainX, trainYd, nh, grupoPorFold);
        end
        
        vetErTrain = [vetErTrain;erroTest];
        vetErVal = [vetErVal;erroVal];
        
        [Ntr,~] = size(testX);

        % Test group
        [Yr,~] = feed_foward([ones(Ntr,1),testX], A, B);
        erroTest = (Yr - testYd);
        erroTest = sum(sum(erroTest.*erroTest))/Ntr;
        vetErTst = [vetErTst;erroTest];
        
        Y{i} = acerto(Yr, testYd);

        % Reset for the next group
        Indices(c) = [];
        i = i+1;
    end
    
    letraYd = paraLetra(testYd);
    letraYr = paraLetra(roundParaConf(Yr));
    [C,order] = confusionmat(letraYd, letraYr);
    
    Y = cell2mat(Y);
    Y = mean(Y);

end

function [path, data, expectedOutput, nh, nkFold] = readInput(input)
    fid = fopen(input, 'r');

    tline = fgetl(fid);
    while ischar(tline)
        
        aux = strsplit(tline, " = ");
        switch aux{1}
            case 'path'
                path = aux{2};
            case 'data'
                data = aux{2};
            case 'expectedOutput'
                expectedOutput = aux{2};
            case 'nh'
                nh = str2num(aux{2});
            case 'nkFold'
                nkFold = str2num(aux{2});
        end
        tline = fgetl(fid);
    end
    fclose(fid);
end

function saveOutput(output, Y, C, order, vetErTrain, vetErVal, vetErTst)
    
    fid = fopen(output,'a');
    fprintf(fid, strcat("Generated at ", datestr(now, 'yyyy-mm-dd HH:MM:SS'), "\n\n"));
    fclose(fid);
    
    saveMSEperFold(output, vetErTrain, vetErVal, vetErTst);
    
    fid = fopen(output,'a');
    
    fprintf(fid,'Average percentage of correct answers: %g\t',Y);
    
    fprintf(fid,'\n\nConfusion matrix\n\n');
    fprintf(fid,'\t');
    transorder = order';
    fprintf(fid,'%c\t',transorder(1,:));
    
    for ii = 1:size(C,1)     
        fprintf(fid,'\n');
        fprintf(fid,'%c\t',order(ii,:));
        fprintf(fid,'%g\t',C(ii,:));
    end
    fclose(fid);
    
    saveAcuracy(output, C, order);
    savePlot(output, vetErTrain, vetErVal, vetErTst);
end

function txAcerto = acerto(Yr,testYd)
    Yr = round(Yr);
    Total = size(Yr,1);
    cont =0;
    for i =1: Total
        mi = testYd(i,:);
        if mi == Yr(i,:)
            cont = cont+1;
        end
    end
    txAcerto = (cont*100)/Total;
end

function log(finished)
    if ~exist('finished', 'var') || ~finished
        tic();
        disp('Running mlp...');
    else
        disp(strcat("Finished in ", string(toc())));
    end
end
