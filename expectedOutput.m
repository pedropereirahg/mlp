function [X, Y] = expectedOutput(expecOutput, path)
    expecOutput = strsplit(expecOutput, " ");
    
    % Check configurations
    if (mod(size(expecOutput, 2), 2) ~= 0)
        return
    end
    
    % Define mapping to generate output
    map = cell(size(expecOutput, 2)/2, 1);
    
    i = 1;
    while i <= size(expecOutput, 2)
        map{round(i/2)} = {expecOutput{i}, str2num(expecOutput{i+1})};
        i = i + 2;
    end
    
    files = dir(path);

    X = cell(size(files));
    Y = cell(size(files,1),1);
    
    i = 1;
    for file = files'
        X{i} = load([file.folder '\' file.name])';
        
        % Put filename on each output mapping
        j = 1;
        while j <= size(map,1)
            map{j}{3} = file.name;
            j = j + 1;
        end
        
        Yaux = cellfun(@setupExpectedOutput, map, 'UniformOutput', false);
        Yaux = Yaux(~cellfun('isempty', Yaux));
        Y{i} = cell2mat(Yaux);

        i = i + 1;
    end
    
    X = cell2mat(X);
    Y = cell2mat(Y);
end

function Y = setupExpectedOutput(prop)
    Y = [];
    if ~isempty(strfind(prop{3}, prop{1}))
        Y = [Y; prop{2}];
    end
end