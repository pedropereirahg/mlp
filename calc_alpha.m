function alfa=calc_alpha(x,d)

alfa_l=0;
alfa_u = rand;
epsilon = 1e-3;

xnew = x + alfa_u*d;
[~,~,g]=calc_grad(xnew);
hl = g'*d;


if abs(hl)<1e-8 %Verifica se hl é zero
    alfa = alfa_u;
    disp('Não realiza busca')
    return 
end

while hl<0 % tenta encontra um alfa que torne h positivo
    alfa_u = 2*alfa_u;     %dobra o valor de alpha
    xnew = x + alfa_u*d;   %calcula o novo valor
    [~,~,g]=calc_grad(xnew); %calcula o valor do gradiente
    hl = g'*d;             %calcula o valor de hl
end

if abs(hl)<1e-8 %Verifica se hl é zero
    alfa = alfa_u;
    disp('Não realiza busca')
    return 
end


nint = ceil (log(alfa_u/epsilon)); % numero de iteações
k=0;

disp(sprintf('***********Buscando um valor de alfa*********************'))
disp(sprintf('Iteração k=%d hl = %1.5f [%1.2f %1.2f]',k, hl, alfa_l,alfa_u))
alfa_m = (alfa_l+alfa_u)/2; % calcula o alfa medio

while k<nint & abs(hl)>1e-8
    k=k+1;
    alfa_m = (alfa_l+alfa_u)/2; % calcula o alfa medio
    xnew = x + alfa_m*d;   %calcula o novo valor
    [~,~,g]=calc_grad(xnew); %calcula o valor do gradiente
    hl = g'*d;              %calcula o valor de hl
    if hl>0
        alfa_u = alfa_m;
    elseif hl <0
        alfa_l = alfa_m;
    else
        break;
    end
    disp(sprintf('Iteração k = %d hl = %1.5f [%1.2f %1.2f]', k, hl, alfa_l,alfa_u))
end
alfa = alfa_m;
disp(sprintf('Iteração k = %d hl = %1.5f melhor alfa =%1.5f ', k, hl, alfa))
disp(sprintf('**********Terminando a busca do alfa**********************'))

