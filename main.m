function main(input, output, verbose)
    
    % Prevent calls without args
    if nargin == 2 && exist('input', 'var') && exist('output', 'var')
    end
        
    if ~exist('input', 'var') || ~exist('output', 'var')
        return
    end
    
    if exist('verbose', 'var') && verbose == true
        t = log();
    end
    
    [path, data, expectedOutput, nh, nkFold] = readInput(input);
    
    % First, try run aplication with loaded data
    % The weights (A,B) will be loaded
    try
        load(data); 
        
        [Y, A, B, C] = runMLP(X, Yd, nh, nkFold, A, B);
        
        saveError(output, Y);
        
        if exist('verbose', 'var') && verbose == true
            log(t);
        end
        clear t  verbose  input  output;
        
        save(data);
        return
    catch
        % Continue
    end

    expectedOutput = strsplit(expectedOutput, " ");
    
    % Check configurations
    if (mod(size(expectedOutput, 2), 2) ~= 0)
        return
    end
    
    % Define mapping to generate output
    map = cell(size(expectedOutput, 2)/2, 1);
    
    i = 1;
    while i <= size(expectedOutput, 2)
        map{round(i/2)} = {expectedOutput{i}, str2num(expectedOutput{i+1})};
        i = i + 2;
    end
    
    files = dir(path);

    X = cell(size(files));
    Yd = cell(size(files,1),1);
    
    i = 1;
    for file = files'
        X{i} = load([file.folder '\' file.name])';
        
        % Put filename on each output mapping
        j = 1;
        while j <= size(map,1)
            map{j}{3} = file.name;
            j = j + 1;
        end
        
        YdAux = cellfun(@setupYd, map, 'UniformOutput', false);
        YdAux = YdAux(~cellfun('isempty', YdAux));
        Yd{i} = cell2mat(YdAux);

        i = i + 1;
    end
    
    X = cell2mat(X);
    Yd = cell2mat(Yd);
    
    % Run the principal loop
    [Y, A, B, C] = runMLP(X, Yd, nh, nkFold);
    
    saveError(output, Y);
    
    if exist('verbose', 'var') && verbose == true
        log(t);
    end
    clear t  verbose  input  output;
    
    save(data);
end

function [Y, A, B, C] = runMLP(X, Yd, nh, nkFold, ALoad, BLoad)
    
    Y = cell(1, nkFold);
    n = size(X,1);
    Indices = 1:n;
    whos
    grupoPorFold = n/nkFold;
    
    i = 1;
    while i <= nkFold
        c = randperm(size(Indices,2), grupoPorFold);  
        [trainX, trainYd, testX, testYd] = kfold(X, Yd, Indices(c));
        
        if exist('ALoad', 'var') && exist('BLoad', 'var')
            [A,B] = mlp(trainX, trainYd, nh, grupoPorFold, ALoad, BLoad);
        else
            [A,B] = mlp(trainX, trainYd, nh, grupoPorFold);
        end

        [Ntr,~] = size(testX);

        %TESTANDO
        [Yr,~] = feed_foward([ones(Ntr,1),testX], A, B);
        Y{i} = acerto(Yr, testYd);

        %RESETRA PRO PROX FOLD
        Indices(c) = [];
        i = i+1;
    end

    letraYd = paraLetra(testYd);
    letraYr = paraLetra(roundParaConf(Yr));
    [C,~] = confusionmat(letraYd, letraYr);
    
    Y = cell2mat(Y);
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

function saveError(output, Y)
    fid = fopen(output,'a');
    
    for ii = 1:size(Y,1)
        fprintf(fid,'%g\t',Y(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
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

function Yd = setupYd(prop)
    Yd = [];
    if ~isempty(strfind(prop{3}, prop{1}))
        Yd = [Yd; prop{2}];
    end
end

function t = log(t)
    if ~exist('t', 'var') 
        t = datetime('now');
        disp('Running mlp...');
    else
        t1 = datetime('now');
        
        disp(strcat("Finished in ", string(between(t, t1))));
        t = t1;
    end
end
