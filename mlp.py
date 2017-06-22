#http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

import numpy as np
import random
from scipy.special import expit
from sklearn.model_selection import KFold


def feed_foward(X,A,B,N):
    Zin = np.dot((np.append(np.ones((N,1)),X,1)),(np.transpose(A)))
    #Z = expit(Zin)
    
    Z = np.float64(1./(1+np.exp(-Zin)))
    Yin = np.dot((np.append(np.ones((N,1)),Z,1)),(np.transpose(B)))
    Y= Yin # Nao tem segunda funcao de ativiacao, arrumar pro EP
    return Y

def gradiente(X,d,A,B,N):
    Zin = np.dot((np.append(np.ones((N,1)),X,1)),(np.transpose(A)))
    Z = np.float64(1./(1+np.exp(-Zin)))
    #Z = expit(Zin)
    Yin = np.dot((np.append(np.ones((N,1)),Z,1)),(np.transpose(B)))
    Y= Yin # Nao tem segunda funcao de ativiacao, arrumar pro EP
    erro = Y - d
    
    grad_aux = np.dot((np.transpose(erro)),(np.append(np.ones((N,1)),Z,1))) 
    # Nao tem segunda funcao de ativiacao, arrumar pro EP
    
    dJdB = (1./N) * (grad_aux)
    dJdZ = np.dot(erro,B)
    dJdZ = np.delete(dJdZ,0,1)
    
    grad_aux = np.dot(np.transpose((dJdZ)*(1-Z)*Z),np.append(np.ones((N,1)),X,1))
    dJdA = (1./N) * grad_aux
    return dJdA,dJdB

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
        alfa_u = 2*alfa_u
        Aaux = A - alfa_u*dJdA;
        Baux = B - alfa_u*dJdB;
        dJdAaux,dJdBaux = gradiente(X,d,Aaux,Baux,N)
        g = vetor_concat(dJdAaux,dJdBaux)
        hl = np.dot(np.transpose(g),dir)

    alfa_m = (alfa_l + alfa_u)/2
    Aaux = A - alfa_u*dJdA;
    Baux = B - alfa_u*dJdB;
    dJdAaux,dJdBaux = gradiente(X,d,Aaux,Baux,N)

    g = vetor_concat(dJdAaux,dJdBaux)
    hl = np.dot(np.transpose(g),dir)
    
    nit = 0;
    nitmax = np.ceil(np.log((alfa_u-alfa_l)/1.0e-5))
    
    while (nit < nitmax and abs(hl)>1.0e-5):
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

def getRight(d,Y, confus = True):
    #print Y
    Y = np.around(Y)
    # print d
    # print Y
    #print(np.array_equal(d[0],Y[0]))
    right =0
    wrong =0
    i = 0
    while(i < len(Y)):
        if(np.array_equal(d[i],Y[i]) is True):
            right = right +1
        else:
            wrong = wrong +1
        i = i+1
    return np.float(right)/len(Y)
#main

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

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
    
d = np.array([[0],
              [0],
              [0],
              [1]])



# X: entrada da rede
# X = np.array([[0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 0],
#               [1, 1],
#               [1, 1],
#               [1, 1],
#               [0, 0],
#               [0, 1],
#               [1, 0]])


# d = np.array([[0],
#               [1],
#               [1],
#               [1],
#               [0],
#               [0],
#               [0],
#               [0],
#               [1],
#               [1]])
# Numero de neuronios da camada do centro
h = 5

# Cria N, ne e ns
# N:  quantidade total de entradas na rede
# ne: tamanho de cada entrada
# ns: tamanho de cada saida
aux = np.shape(X)
N = aux[0]
ne = aux[1]
aux = np.shape(d)
#print (np.shape(X))
ns = aux[1]

# Cria os pesos aleatorios para A e B
A = np.random.rand(h,(ne+1))
B = np.random.rand(ns,(h+1))

#print A
#print B

# Feedfoward para a saida
Y = feed_foward(X,A,B,N)
erro = Y - d
EQM = (1./N) * ((erro*erro).sum())

i = 0
alfa = 1

vEQM = []
vEQM.append(EQM)

# ANTES DE TREINAR
#print(Y)
#print(map(round, Y))
#print(map(round, d))
#print(EQM)

# TREINAMENTO
while (EQM > 1.0e-5 and i<10000):
    i = i+1
    dJdA,dJdB = gradiente(X,d,A,B,N)
    alfa = bis_mlp(X,d,A,B,dJdA,dJdB,N)
    A = A - alfa*dJdA
    B = B - alfa*dJdB
    Y = feed_foward(X,A,B,N)
    erro = Y - d
    EQM = (1./N) * ((erro*erro).sum())
    print EQM
    vEQM.append(EQM)
Y = feed_foward(X,A,B,N)

aux = getRight(d,Y)
print Y
print aux
#print(Y)
print(map(round, Y))
print(map(round, d))
#print(EQM)

print i

#print(np.exp(7))
    #para
    # np.seterr(all='raise', over='raise')
    
    # rra = np.array([
    # [-175.88811963,-525.74348929,-229.68880417],
    # [-258.15581499, -431.27570672 ,-326.45780328],
    # [-592.7936927  ,-156.27748283 ,-354.65599997],
    # [-339.79424678, -367.27805576 ,-335.48888291],
    # [-614.50264703 ,-156.5383433  ,-291.11893938],
    # [-243.12799529 ,-453.04785808 ,-271.08139235],
    # [-613.15320377 ,-162.35495486 ,-280.81062755],
    # [ -85.60601923 ,-701.77760356 , -18.76827397],
    # [ -69.67967026 ,-713.81754045 , -41.82732041],
    # [-117.14711921 ,-655.53087108 , -74.59603325],
    # [ -83.78568558 ,-713.30713739 , -51.98939519],
    # [ -18.12784592 ,-756.59140873 ,-7.81847052],
    # [-599.40995641 ,-154.54729834 ,-351.23481571],
    # [-130.07759095 ,-566.97217148 ,-237.01500963],
    # [-210.57571369 ,-555.68496898 ,-112.85697897],
    # [-536.94555967 ,-194.44348991 ,-331.19195789],
    # [-274.26230065 ,-437.9916248  ,-257.96252686],
    # [-68.06432329 ,-674.09537164  ,-99.88289222]])
    
    # teste = np.float64(1./(1+np.exp(-rra)))
    
    # try:
    #     teste = np.float64(1./(1+np.exp(-rra)))
    # except FloatingPointError:
    #     print("teste")
        
    #print(np.exp(-614))