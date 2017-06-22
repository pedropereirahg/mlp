import numpy as np
import os
import random
from test2 import mlp
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def cria_val(tgroup,val_size):
    it = 0
    a =-1
    vg = []
    while it < val_size:
        new=random.randint(0,(tgroup.size-1))
        if(new != a):
            vg.append(tgroup[new])
            a = new
            it = it+1
            tgroup=np.delete(tgroup,new)
    it = 1
    return tgroup,vg 
    
def montaXd(auxTr,auX,d):
    new_X =[]
    new_d =[]
    for x in auxTr:
        new_X.append(auX[x])
        new_d.append(d[x])
    new_X= np.reshape(new_X, (-1, auX[0].size))
    new_d= np.reshape(new_d, (-1, d[0].size))
    return new_X,new_d

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

if __name__ == '__main__':
    
    # Construct X from files on build folder
    # Please, first choose the descriptor
    
    path = "build/hog/ppc_32_cpb_1_o_9/"
    
    # Matriz de entrada
    X = []
    # Matriz de saida esperada
    d = []
    
    os.chdir(path + "images/")
    
    for content in os.listdir("."):
        
        current_dir = os.getcwd() + "/"
        
        vet_aux = []
        
        f = open(current_dir + content, 'r')
        lines = f.read()
        lines = lines.split("\n")[1:]
        for line in lines:
            if not line:
                continue
            vet_aux.append(np.float(line))
        f.close()
        
        vet_aux = np.array(vet_aux)
        X.append(vet_aux)
        
        
        # Letra Z
        if ("5a" in content):
            aux_d = map(np.float, np.array([0, 0, 1]))
            #d.append(map(np.float, np.array([0, 0, 1])))
            d.append(np.array(aux_d))
        # Letra S
        if ("53" in content):
            aux_d = map(np.float, np.array([0, 1, 0]))
            #d.append(map(np.float, np.array([0, 1, 0])))
            d.append(np.array(aux_d))
        # Letra X
        if ("58" in content):
            aux_d = map(np.float, np.array([1, 0, 0]))
           #d.append(map(np.float, np.array([1, 0, 0])))
            d.append(np.array(aux_d))
            
        


    # Numero de neuronios
    h = 50
    
    #print(d)
    
    X = np.reshape(X, (-1, X[0].size))
    d = np.reshape(d, (-1, d[0].size))
   
    #TEMPORARIO RETIRAR XOR
    # X = np.array([[0, 0],
    #           [0, 1],
    #           [1, 0],
    #           [1, 0],
    #           [1, 1],
    #           [1, 1],
    #           [1, 1],
    #           [0, 0],
    #           [0, 1],
    #           [1, 0]])
    
    # d = np.array([[0],
    #           [1],
    #           [1],
    #           [1],
    #           [0],
    #           [0],
    #           [0],
    #           [0],
    #           [1],
    #           [1]])
    
    n_splits=5
    kf = KFold(n_splits)
    erro_total = []
    for train_group, test_group in kf.split(X):
        
        # Grupo de treinamento
        train_group,val_group = cria_val(train_group, test_group.size)
        
        tr_X,tr_d = montaXd(train_group,X,d)
        val_x,val_d = montaXd(val_group,X,d)
        tst_x,tst_d = montaXd(test_group,X,d)
        
        Y, A, B, EQM_tr, vEQM = mlp(tr_X, tr_d,h,val_x,val_d)
    
        Y,EQM_tst = mlp(tst_x,tst_d,h,tst_x,tst_d,False, A,B)
        
        err_porc = getRight(tst_d,Y)
        erro_total.append(err_porc)
        
        print Y
        print tst_d
        #print("saida          " + str(map(round, Y)))
        #print("saida esperada " + str(map(round, tst_d)))
        #print("error          " + str(EQM_tst))
        # print("\n")

    #Y, A, B, EQM, vEQM = mlp(X,d,h)
    
    print(erro_total)
    print(np.average(erro_total))
    # print("saida          " + str(map(round, Y)))
    # print("saida esperada " + str(map(round, d)))
    # print("error          " + str(EQM))
    # print(vEQM)
    # print("\n\n")