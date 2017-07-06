function [X, Y, map] = expectedOutput(expecOutput, path)
    expecOutput = strsplit(expecOutput, " ");
    
    % Check configurations
    if (mod(size(expecOutput, 2), 2) ~= 0) 
        error("expectedOutput doesn't multiple of 2.")
    end
    
    % Define mapping to generate output
    map = cell(size(expecOutput, 2)/2, 1);
    
    i = 1;
    while i <= size(expecOutput, 2)
        map{round(i/2)} = {expecOutput{i}, str2num(expecOutput{i+1})};
        i = i + 2;
    end
    
    try
        [X, Y] = readExpectedOutput();
        X = cell2mat(X);
        Y = cell2mat(Y);
        return
    catch
        % continue...
    end
    
    files = dir([path '*.txt']);

    X = cell(size(files));
    Y = cell(size(files,1),1);
    mapCopy = map;
    
    i = 1;
    for file = files'
        X{i} = load([path file.name])';
        
        % Put filename on each output mapping
        j = 1;
        while j <= size(mapCopy,1)
            mapCopy{j}{3} = file.name;
            j = j + 1;
        end
        
        Yaux = cellfun(@setupExpectedOutput, mapCopy, 'UniformOutput', false);
        Yaux = Yaux(~cellfun('isempty', Yaux));
        Y{i} = cell2mat(Yaux);

        i = i + 1;
    end
    
    try
        saveExpectedOutput(X,Y);
    catch
        % continue...
    end
    
    X = cell2mat(X);
    Y = cell2mat(Y);
end

function [X, Y] = readExpectedOutput()
    X = readtable('expectedOutputX.txt','Delimiter',' ','ReadVariableNames',false);
    X = table2cell(X);
    Y = readtable('expectedOutputYd.txt','Delimiter',' ','ReadVariableNames',false);
    Y = table2cell(Y);
end

function saveExpectedOutput(X,Y)
    Tx = cell2table(X);
    writetable(Tx,'expectedOutputX.txt','Delimiter',' ');
    Ty = cell2table(Y);
    writetable(Ty,'expectedOutputYd.txt','Delimiter',' ');
end

function Y = setupExpectedOutput(prop)
    Y = [];
    if ~isempty(strfind(prop{3}, prop{1}))
        Y = [Y; prop{2}];
    end
end