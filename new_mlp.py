#http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

import numpy as np
from scipy.special import expit



def feed_foward(X,A,B,N):
   
    Zin = np.dot((np.append(np.ones((N,1)),X,1)),(np.transpose(A)))
    #print Zin
    #print ("ZINFeed")
    Z = np.float64(1./(1+np.exp(-Zin)))
    Yin = np.dot((np.append(np.ones((N,1)),Z,1)),(np.transpose(B)))
    #Y= expit(Yin) # Nao tem segunda funcao de ativiacao, arrumar pro EP
    #print ("YINFeed")
    #print Yin
    Y = np.float64(1./(1+np.exp(-Yin)))
    
    Y = np.reshape(Y, (-1,1),'F')
    return Y

def gradiente(X,d,A,B,N):
    Zin = np.dot((np.append(np.ones((N,1)),X,1)),(np.transpose(A)))
    #print ("ZINGrad")
    #print Zin
    Zin= np.around(Zin,decimals=4)
    Z = np.float64(1./(1+np.exp(-Zin)))
    #Z = expit(Zin)
    Yin = np.dot((np.append(np.ones((N,1)),Z,1)),(np.transpose(B)))
    #print ("YINGrad")
    #print Yin
    Yin= np.around(Yin,decimals=4)
    Y = np.float64(1./(1+np.exp(-Yin)))
    #Y=Yin
    Y= np.reshape(Y, (-1,1),'F')
    erro = Y - d
   
    #grad_aux = np.dot((np.transpose(erro)),(np.append(np.ones((N,1)),Z,1))) 
    #aux = np.transpose(erro*((1 - Y)*Y))
    grad_aux = np.dot((np.transpose(erro*((1 - Y)*Y))),(np.append(np.ones((N, 1)), Z, 1)))
    
    dJdB = (1./N) * (grad_aux)
 
    
    #dJdZ = (erro*B)
    dJdZ = np.dot(erro,B)
    #print dJdZ
    dJdZ = np.delete(dJdZ,0,1)
    grad_aux = np.dot(np.transpose((dJdZ)*(1-Z)*Z),np.append(np.ones((N,1)),X,1))
    #grad_aux = np.dot(np.transpose(dJdZ*(1-Z)*Z), np.append(np.ones((N, 1)), X, 1))
    dJdA = (1./N) * grad_aux
    
    #print dJdA
    #print dJdB
    return dJdA,dJdB
    #return np.around(dJdA,decimals=4),np.around(dJdB,decimals=4)

def vetor_concat(a,b):
    a = np.reshape(a, (-1,1),'F')
    b = np.reshape(b, (-1,1),'F')
    aux = np.concatenate((a, b), axis=0)
    return aux
    
def bis_mlp(X,d,A,B,dJdA,dJdB,N):
    dir = vetor_concat(-dJdA,-dJdB)
    
    alfa_l = 0
    alfa_u = np.random.uniform(0,1,1)
    
    Aaux = A - alfa_u*dJdA
    Baux = B - alfa_u*dJdB
    
    dJdAaux,dJdBaux = gradiente(X,d,Aaux,Baux,N)
    
    
    g = vetor_concat(dJdAaux,dJdBaux)
    hl = np.dot(np.transpose(g),dir)
    
    
    while (hl<0):
        #print("HERE")
        alfa_u = 2*alfa_u
        Aaux = A - alfa_u*dJdA;
        Baux = B - alfa_u*dJdB;
        dJdAaux,dJdBaux = gradiente(X,d,Aaux,Baux,N)
        #print dJdAaux
        #print dJdBaux
        #print("\n")
        g = vetor_concat(dJdAaux,dJdBaux)
        hl = np.dot(np.transpose(g),dir)
   
    #para
    alfa_m = (alfa_l + alfa_u)/2
    Aaux = A - alfa_u*dJdA;
    Baux = B - alfa_u*dJdB;
    dJdAaux,dJdBaux = gradiente(X,d,Aaux,Baux,N)

    g = vetor_concat(dJdAaux,dJdBaux)
    hl = np.dot(np.transpose(g),dir)
    
    nit = 0;
    nitmax = np.ceil(np.log((alfa_u-alfa_l)/1.0e-5))
    
    while (nit < nitmax and abs(hl)>1.0e-5):
        #print("AQUI")
        nit = nit +1
        if (hl>0):
            alfa_u = alfa_m
        else:
            alfa_l = alfa_m
        alfa_m = (alfa_l+alfa_u)/2;
        Aaux = A - alfa_m*dJdA;
        Baux = B - alfa_m*dJdB;   
        dJdAaux,dJdBaux = gradiente(X,d,Aaux,Baux,N)
        g = vetor_concat(dJdAaux,dJdBaux)
        hl = np.dot(np.transpose(g),dir)
    alfa = alfa_m
    return alfa

def over(type,flag):
    
    print(type)

def mlp(X, d, h, train = True, *args, **kwargs):
    #print(np.around(-0.00042509,decimals=4))   
    #stop
   
    np.seterr(all='raise', over='raise')
    np.seterrcall(over)
    
    # X = np.array([[0, 0],
    #               [0, 1],
    #               [1, 0],
    #               [1, 1]])
    
    # d = np.array([[0],
    #               [1],
    #               [1],
    #               [0]])
    
    
    # Pega argumentos opcionais passados
    A = kwargs.get('A', None)
    B = kwargs.get('B', None)
    val_x = kwargs.get('val_x', None)
    val_d = kwargs.get('val_d', None)
    
    # Cria N, ne e ns
    # N:  quantidade total de entradas na rede
    # ne: tamanho de cada entrada
    # ns: tamanho de cada saida
    aux = np.shape(X)
    N = aux[0]
    ne = aux[1]
    aux = np.shape(d)
    ns = aux[1]
    
    # Cria os pesos aleatorios para A e B caso nao exista
    if (A is None):
        A = np.random.rand(h,(ne+1))
    if (B is None):
        B = np.random.rand(ns,(h+1))
    
    #BASICAO PRO MATLAB
    #A = np.array([[-0.5649,0.40644,-0.57593],[-0.49791,0.11147,-0.84530],[0.78584,-0.63113,0.82760]])
    #B = np.array([0.4134,0.1156,-0.3731,-0.6676])
   
    #RESPOSTA GET STUCK
    A = np.array([[ 0.83844721,0.60360501,0.74588541],[ 0.98090387,0.26572375,0.37104968],[ 0.98878775,0.43045012,0.09037516]])
    B= np.array([[0.03476854,0.99209828, 0.11645143 ,0.12159723]])
    
    # Feedfoward para a saida
    Y = feed_foward(X,A,B,N)

    error = Y - d
    avg_error = (1./N) * ((error*error).sum()) # ERRO QUADRADO MEDIO
    vet_errors = []
    vet_errors.append(avg_error)
    
    if (train == False):
        return [Y, avg_error]
    
    #Y, error1 = mlp(val_x, val_d, h, False, A=A, B=B)
    
    alfa = 1
    i=0
    
    Ain= A
    Bin =B
    while (avg_error > 1.0e-5 and i <1000):
        i=i+1
        dJdA,dJdB = gradiente(X,d,A,B,N)
        alfa = bis_mlp(X,d,A,B,dJdA,dJdB,N)
        A = A - alfa*dJdA
        B = B - alfa*dJdB
        Y = feed_foward(X,A,B,N)
        error = Y - d
        avg_error = (1./N) * ((error*error).sum())
        print dJdA
        #Y,error2 = mlp(val_x, val_d, h, False, A=A, B=B)
       # print error1,error2
        # if (error2 < error1):
        #     i=0
        #     error1 = error2
        #     Abom = A
        #     Bbom = B
        # else:
        #     i=i+1
    #print np.around(Y,decimals=4)
    print("\n")
    print Y
    print("\n") 
    print Ain
    print Bin
    para
    return [Y, Abom, Bbom, np.sum(vet_errors), vet_errors]


if __name__ == '__main__':
        
    # O PROBLEMA: XOR
    # O xor eh uma operacao logica entre dois operandos que resulta em um valor 
    # logico verdadeiro se e somente se exatamente UM dos operandos possui valor verdadeiro.
    # 
    # Exemplo 1: 
    #   Entrada 0, 0
    #   Saida 0
    # 
    # Exemplo 2: 
    #   Entrada 0, 1
    #   Saida 1
    
    
    
    # X: entrada da rede
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 0],
                  [1, 1],
                  [1, 1],
                  [1, 1],
                  [0, 0],
                  [0, 1],
                  [1, 0]])
    
    # [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # Saida esperada da rede
    d = np.array([[0],
                  [1],
                  [1],
                  [1],
                  [0],
                  [0],
                  [0],
                  [0],
                  [1],
                  [1]])
    # Numero de neuronios
    h = 2
    
    Y, EQM = mlp(X, d, h, False)
    
    print("saida          " + str(map(round, Y)))
    print("saida esperada " + str(map(round, d)))
    print("error          " + str(EQM))
    print("\n\n")
    
    Y, A, B, EQM, vEQM = mlp(X, d, h, True, val_x=X, val_d=d)
    
    print("saida          " + str(map(round, Y)))
    print("saida esperada " + str(map(round, d)))
    print("error          " + str(EQM))
    print("\n\n")
    
    
    Y, EQM = mlp(X, d, h, False, A=A, B=B)
    
    print("saida          " + str(map(round, Y)))
    print("saida esperada " + str(map(round, d)))
    print("error          " + str(EQM))
    print("\n\n")
