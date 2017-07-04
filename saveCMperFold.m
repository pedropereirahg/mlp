function saveCMperFold(output, cVet, orderVet)
    fid = fopen(output,'a');
    
    for i = 1:size(cVet, 2)
        fprintf(fid,['\n\nConfusion matrix: fold ', num2str(i), '\n\n']);
        fprintf(fid,'\t');
        transorder = orderVet{i}';
        fprintf(fid,'%c\t', transorder(1,:));

        for ii = 1:size(cVet{i},1)
            fprintf(fid,'\n');
            fprintf(fid,'%c\t',orderVet{i}(ii,:));
            fprintf(fid,'%g\t',cVet{i}(ii,:));
        end
        
        saveAcuracy(output, cVet{i}, orderVet{i});
    end
    fprintf(fid,'\n');
    
    fclose(fid);
end