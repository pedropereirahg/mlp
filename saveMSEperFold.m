function saveMSEperFold(output,eqmTr,eqmVal,eqmTs)
    fid = fopen(output,'a');
    
    fprintf(fid,'Mean square error\n');
    fprintf(fid,'\n');

    for i = 1:size(eqmTr,1)
        fprintf(fid,'%u;',i);
        fprintf(fid,'%g;',eqmTr(i));
        fprintf(fid,'%g;',eqmVal(i));
        fprintf(fid,'%g\t',eqmTs(i));
        fprintf(fid,'\n');
    end
    
    fclose(fid);
end