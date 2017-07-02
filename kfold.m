function [TRX,TRY,TSTX,TSTY] = kfold(X,Yd,c)
    TSTX=X(c,:); 
    TSTY=Yd(c,:);

    TRX = X;
    TRX(c,:)= [];
    TRY = Yd;
    TRY(c,:) = [];
end