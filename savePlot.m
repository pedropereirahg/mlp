function savePlot(output, vetErTrain, vetErVal, vetErTst)
    output = strsplit(output, ".");
    gcf = figure('visible', 'off');
    plot(vetErTrain);
    saveas(gcf, [output{1}, '_train'], 'png');
    plot(vetErVal);
    saveas(gcf, [output{1}, '_valid'], 'png');
    plot(vetErTst);
    saveas(gcf, [output{1}, '_tests'], 'png');
end