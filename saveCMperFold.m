function saveCMperFold(output, cVet, orderVet)
    fid = fopen(output,'a');
    
    for i = 1:size(cVet, 2)
        fprintf(fid,['\n\nConfusion matrix: fold ', num2str(i), '\n\n']);
        fprintf(fid,'\t\t');
        transorder = orderVet{i}';
        fprintf(fid,'%s\t', transorder{1,:});

        for ii = 1:size(cVet{i},1)
            fprintf(fid,'\n');
            fprintf(fid,'%s\t',orderVet{i}{ii,1});
            fprintf(fid,'%d\t\t',cVet{i}(ii,:));
        end
        fclose(fid);
        fid = fopen(output,'a');
        
        saveAcuracy(output, cVet{i}, orderVet{i});
    end
    fprintf(fid,'\n');
    
    fclose(fid);
end