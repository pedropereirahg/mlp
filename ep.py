import numpy as np
import os
import random
from new_mlp import mlp
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def validation(group, size):
    #print group
    # Guarda o grupo original antes de retirar os elementos sorteados
    aux_group = np.copy(group)
    
    # Sorteia elementos a partir do grupo de validacao
    if group.size ==size:
        size = size/2
    val_indexes = random.sample(range(group.size), size)
    #print group.size
    # Inicializa o grupo de valicao
    val_group = np.array([], dtype=np.int64)
    
    # Retira os elementos sorteados do grupo original
    group = np.delete(group, val_indexes)
    
    # Determina o grupo de validacao a patir dos indices do grupo sorteado
    val_group = np.append(val_group, [aux_group[i] for i in val_indexes])
    
    #print group, val_group
    return group, val_group
    
def montaXd(auxTr,auX,d):
    new_X =[]
    new_d =[]
    for x in auxTr:
        new_X.append(auX[x])
        new_d.append(d[x])
    new_X= np.reshape(new_X, (-1, auX[0].size))
    new_d= np.reshape(new_d, (-1, d[0].size))
    return new_X,new_d

def getRight(d,Y):
    print Y
    print d
    Y = np.around(Y)
    print Y
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
    
    # Numero de neuronios
    h = 3
    
    # Numerod de grupos do kFold
    n_splits = 5
    
    # Teste com xor
    is_xor = False
    
    if (is_xor):
        os.chdir(path + "images_xor/")
    else:
        os.chdir(path + "images/")
    
    for content in os.listdir("."):
        
        current_dir = os.getcwd() + "/"
        
        vet_aux = []
        
        f = open(current_dir + content, 'r')
        lines = f.read()
        lines = lines.split("\n")
        for line in lines:
            if not line:
                continue
            vet_aux.append(np.float(line))
        f.close()
        
        X.append(np.array(vet_aux))
        #print d
        if (is_xor):
            # Resultado 1
            if ("true" in content):
                d.append(np.array(map(np.float, np.array([1]))))
            # Resultado 0
            if ("false" in content):
                d.append(np.array(map(np.float, np.array([0]))))
        else:
            # Letra Z
            if ("5a" in content):
                d.append(np.array(map(np.float, np.array([[0, 0, 1]]))))
            # Letra S
            if ("53" in content):
                d.append(np.array(map(np.float, np.array([[0, 1, 0]]))))
            # Letra X
            if ("58" in content):
                d.append(np.array(map(np.float, np.array([[1, 0, 0]]))))
            

    X = np.reshape(X, (-1, X[0].size))
    d = np.reshape(d, (-1, d[0].size))
    
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
    
    
    kf = KFold(n_splits)
    erro_total = []
    for train_group, test_group in kf.split(X):
        
        #print train_group, test_group
        # Cria grupo de validacao a partir do
        # grupo de treinamento
        train_group, val_group = validation(train_group, test_group.size)
        
        tr_X, tr_d = montaXd(train_group,X,d)
        val_x, val_d = montaXd(val_group,X,d)
        tst_x, tst_d = montaXd(test_group,X,d)
        
        Y, A, B, EQM_tr, vEQM = mlp(tr_X, tr_d, h, True, val_x=val_x, val_d=val_d)
        
        Y, EQM_tst = mlp(tst_x, tst_d, h, False, A=A, B=B)
        
        err_porc = getRight(tst_d,Y)
        erro_total.append(err_porc)
        
    #Y, A, B, EQM, vEQM = mlp(X,d,h)
    
    print(erro_total)
    print(np.average(erro_total))
    