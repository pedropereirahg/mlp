function saveAcuracy(output, C, order)
    fid = fopen(output,'a');
    
    fprintf(fid,'\n\nAcuracy\n');
    fprintf(fid,'\n');

    for i = 1:size(C,1)     
        fprintf(fid,'%s\t',order{i,1});
        Ac = (C(i,i))/(sum(C(i,:)));
        fprintf(fid,'%g\t',Ac);
        fprintf(fid,'\n');
    end
    fprintf(fid,'\n');
    
    fclose(fid);
end
