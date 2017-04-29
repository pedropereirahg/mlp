###################################
#   Multiplica-se os dados de entradas com os pesos A
#   Aplica-se a funcao de ativacao f
#   Multiplica-se os dados da camada escondida com os pesos B
#   Aplica-se a funcao de ativacao g
#   --------BACKPROPAGATION----------
#   Calculamos o erro baseado na saida
#   Multiplicamos o erro
#   Soma-se as camadas escondidas
#   Substituimos o valor na derivada f'
#
#######################################
#http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

import numpy as np
from scipy.special import expit
#import bigfloat
#bigfloat.exp(5000,bigfloat.precision(100))

def feed_foward(X,A,B,N):
    Zin = np.dot((np.append(np.ones((N,1)),X,1)),(np.transpose(A)))
    #Zin = np.float64(Zin)
    #Z = expit(Zin)
    Z = np.float64(1./(1+np.exp(-Zin)))
    Yin = np.dot((np.append(np.ones((N,1)),Z,1)),(np.transpose(B)))
    Y= Yin # Nao tem segunda funcao de ativiacao, arrumar pro EP
    #print (Y)
    return Y

def gradiente(X,d,A,B,N):
    Zin = np.dot((np.append(np.ones((N,1)),X,1)),(np.transpose(A)))
    Z = np.float64(1./(1+np.exp(-Zin)))
    #Z = expit(Zin)
    Yin = np.dot((np.append(np.ones((N,1)),Z,1)),(np.transpose(B)))
    Y= Yin # Nao tem segunda funcao de ativiacao, arrumar pro EP
    erro = Y - d
    
    grad_aux = np.dot((np.transpose(erro)),(np.append(np.ones((N,1)),Z,1))) # Nao tem segunda funcao de ativiacao, arrumar pro EP
    
    dJdB = (1./N) * (grad_aux)
    dJdZ = np.dot(erro,B)
    dJdZ = np.delete(dJdZ,0,1)
    
    grad_aux = np.dot(np.transpose((dJdZ)*(1-Z)*Z),np.append(np.ones((N,1)),X,1))
    dJdA = (1./N) * grad_aux
    return dJdA,dJdB

def vetor_concat(a,b):
    a = np.reshape(a, (-1,1),'F')
    b = np.reshape(a, (-1,1),'F')
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
    #print(hl)
    while (hl<0):
        #print(hl)
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

#Seta X,d ,H
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]) #Setar o X
d = np.array([[0],[1],[1],[0]])
h = 3

# Cria N, ne e ns
aux = np.shape(X)
N = aux[0]
ne = aux[1]
aux = np.shape(d)

ns = aux[1]

#Cria os pesos aleatorios para A e B
A = np.random.rand(h,(ne+1))
B = np.random.rand(ns,(h+1))
#print(A)

#Feedfoward para a saida
Y = feed_foward(X,A,B,N)
erro = Y - d
EQM = (1./N) * ((erro*erro).sum())
i = 0
alfa = 1

vEQM = []
vEQM.append(EQM)

while (EQM > 1.0e-5 and i<100):
    i = i+1
    dJdA,dJdB = gradiente(X,d,A,B,N)
    alfa = bis_mlp(X,d,A,B,dJdA,dJdB,N)
    A = A - alfa*dJdA
    B = B - alfa*dJdB
    Y = feed_foward(X,A,B,N)
    erro = Y - d
    EQM = (1./N) * ((erro*erro).sum())
    #print(Y)
Y = feed_foward(X,A,B,N)
#print("deu merda")
print(Y)