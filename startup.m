arg_list = argv();
if size(arg_list, 1) >= 2
    inputFile = arg_list{1};
    outputFil = arg_list{2};
end

if size(arg_list, 1) > 2
    verbose = arg_list{3};
    main(inputFile, outputFil, verbose);
else 
    main(inputFile, outputFil);
end


