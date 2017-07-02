function alfa = bissecao_mlp(X,d,A,B,dJdA,dJdB,N)
    dir = [-dJdA(:);-dJdB(:)];

    alfa_l = 0;
    alfa_u = rand(1,1);
    Aaux = A - alfa_u*dJdA;
    Baux = B - alfa_u*dJdB;
    [dJdAaux,dJdBaux] = gradient(X, d, Aaux, Baux, N);
    g = [dJdAaux(:);dJdBaux(:)];
    hl = g'*dir;
    while hl<0
        alfa_u = 2*alfa_u;
        Aaux = A - alfa_u*dJdA;
        Baux = B - alfa_u*dJdB;
        [dJdAaux,dJdBaux] = gradient(X, d, Aaux, Baux, N);
        g = [dJdAaux(:);dJdBaux(:)];
        hl = g'*dir;
    end

    alfa_m = (alfa_l+alfa_u)/2;
    Aaux = A - alfa_m*dJdA;
    Baux = B - alfa_m*dJdB;
    [dJdAaux,dJdBaux] = gradient(X,d,Aaux,Baux,N);
    g = [dJdAaux(:);dJdBaux(:)];
    hl = g'*dir;

    nit = 0;
    nitmax = ceil(log((alfa_u-alfa_l)/1.0e-5));

    while nit<nitmax && abs(hl)>1.0e-5
        nit = nit+1;
        if hl>0
            alfa_u = alfa_m;
        else
            alfa_l = alfa_m;
        end 
        alfa_m = (alfa_l+alfa_u)/2;
        Aaux = A - alfa_m*dJdA;
        Baux = B - alfa_m*dJdB;
        [dJdAaux,dJdBaux] = gradient(X, d, Aaux, Baux, N);
        g = [dJdAaux(:);dJdBaux(:)];
        hl = g'*dir;
    end
    alfa = alfa_m;