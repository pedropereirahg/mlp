# http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

import numpy as np
import os
from scipy.special import expit
from sklearn.model_selection import KFold


def configs(path):
    configs = {}

    # defaults
    mlp_h = 3
    mlp_ns = 3
    iter_max = 1000
    alfa = 1
    mlp_letter_z = [0, 1, 0]
    mlp_letter_s = [0, 1, 0]
    mlp_letter_x = [1, 0, 0]

    f = open(path, 'r')
    lines = f.read()
    lines = lines.split("\n")[1:]
    for line in lines:
        if not line:
            continue
        p = line.split(" : ")
        if "HOG" in p[0] or "LENGTH" in p[0]:
            configs[p[0]] = int(p[1])
        else:
            configs[p[0]] = p[1]
    f.close()

    # Set H default
    if "MLP_H" in configs:
        configs["MLP_H"] = int(configs["MLP_H"])
    else:
        f = open(path, 'a')
        f.write("MLP_H : %s\n" % str(mlp_h))
        f.close()
        configs["MLP_H"] = mlp_h

    # Set NS default
    if "MLP_NS" in configs:
        configs["MLP_NS"] = int(configs["MLP_NS"])
    else:
        f = open(path, 'a')
        f.write("MLP_NS : %s\n" % str(mlp_ns))
        f.close()
        configs["MLP_NS"] = mlp_ns

    # Set ITER_MAX default
    if "MLP_ITER_MAX" in configs:
        configs["MLP_ITER_MAX"] = int(configs["MLP_ITER_MAX"])
    else:
        f = open(path, 'a')
        f.write("MLP_ITER_MAX : %s\n" % str(iter_max))
        f.close()
        configs["MLP_ITER_MAX"] = iter_max

    # Set ALFA default
    if "MLP_ALFA" in configs:
        configs["MLP_ALFA"] = float(configs["MLP_ALFA"])
    else:
        f = open(path, 'a')
        f.write("MLP_ALFA : %s\n" % str(alfa))
        f.close()
        configs["MLP_ALFA"] = alfa

    # Set LETTER_Z default
    if "MLP_LETTER_Z" in configs:
        configs["MLP_LETTER_Z"] = map(int, configs["MLP_LETTER_Z"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_LETTER_Z : %s\n" % ",".join(map(str, mlp_letter_z)))
        f.close()
        configs["MLP_LETTER_Z"] = mlp_letter_z

    # Set LETTER_S default
    if "MLP_LETTER_S" in configs:
        configs["MLP_LETTER_S"] = map(int, configs["MLP_LETTER_S"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_LETTER_S : %s\n" % ",".join(map(str, mlp_letter_s)))
        f.close()
        configs["MLP_LETTER_S"] = mlp_letter_s

    # Set LETTER_X default
    if "MLP_LETTER_X" in configs:
        configs["MLP_LETTER_X"] = map(int, configs["MLP_LETTER_X"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_LETTER_X : %s\n" % ",".join(map(str, mlp_letter_x)))
        f.close()
        configs["MLP_LETTER_X"] = mlp_letter_x

    # Firstly that is random
    mlp_a = np.random.rand(configs["MLP_H"],  (configs["MLP_X_LENGTH"] + 1))
    mlp_b = np.random.rand(configs["MLP_NS"], (configs["MLP_H"] + 1))
    average_error = []

    # Set A default
    if "MLP_A" in configs:
        configs["MLP_A"] = [x.split(",") for x in configs["MLP_A"].split(";")]
        configs["MLP_A"] = np.float_(configs["MLP_A"])
    else:
        f = open(path, 'a')
        f.write("MLP_A : %s\n" % ';'.join(','.join('%f' % x for x in y) for y in mlp_a))
        f.close()
        configs["MLP_A"] = mlp_a

    # Set B default
    if "MLP_B" in configs:
        configs["MLP_B"] = [x.split(",") for x in configs["MLP_B"].split(";")]
        configs["MLP_B"] = np.float_(configs["MLP_B"])
    else:
        f = open(path, 'a')
        f.write("MLP_B : %s\n" % ';'.join(','.join('%f' % x for x in y) for y in mlp_b))
        f.close()
        configs["MLP_B"] = mlp_b

    # Set AVERAGE_ERROR default
    if "MLP_AVERAGE_ERROR" in configs:
        configs["MLP_AVERAGE_ERROR"] = map(float, configs["MLP_AVERAGE_ERROR"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_AVERAGE_ERROR : %s\n" % ",".join(map(str, average_error)))
        f.close()
        configs["MLP_AVERAGE_ERROR"] = average_error

    # Set AVERAGE_ERROR_TEST default
    if "MLP_AVERAGE_ERROR_TEST" in configs:
        configs["MLP_AVERAGE_ERROR_TEST"] = map(float, configs["MLP_AVERAGE_ERROR_TEST"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_AVERAGE_ERROR_TEST : %s\n" % ",".join(map(str, average_error)))
        f.close()
        configs["MLP_AVERAGE_ERROR_TEST"] = average_error

    return configs


def feed_forward(x, a, b, n):
    z_in = np.dot((np.append(np.ones((n, 1)), x, 1)), (np.transpose(a)))
    z = expit(z_in)
    y_in = np.dot((np.append(np.ones((n, 1)), z, 1)), (np.transpose(b)))
    y = expit(y_in)  # Nao tem segunda funcao de ativiacao, arrumar pro EP
    return y


def gradient(x, d, a, b, n):
    z_in = np.dot((np.append(np.ones((n, 1)), x, 1)), (np.transpose(a)))
    z = expit(z_in)
    y_in = np.dot((np.append(np.ones((n, 1)), z, 1)), (np.transpose(b)))
    y = expit(y_in)  # Nao tem segunda funcao de ativiacao, arrumar pro EP
    error = y - d

    grad_aux = np.dot((np.transpose(error * ((1 - y) * y))),
                      (np.append(np.ones((n, 1)), z, 1)))  # Nao tem segunda funcao de ativiacao, arrumar pro EP

    d_jd_b = (1. / n) * grad_aux

    d_jd_z = np.dot(error, b)
    d_jd_z = np.delete(d_jd_z, 0, 1)

    grad_aux = np.dot(np.transpose(d_jd_z * (1 - z) * z), np.append(np.ones((n, 1)), x, 1))
    d_jd_a = (1. / n) * grad_aux
    return d_jd_a, d_jd_b


def vet_concat(a, b):
    a = np.reshape(a, (-1, 1), 'F')
    b = np.reshape(b, (-1, 1), 'F')
    return np.concatenate((a, b), axis=0)


def bis_mlp(x, d, a, b, d_jd_a, d_jd_b, n):
    dir = vet_concat(-d_jd_a, -d_jd_b)

    alfa_l = 0
    alfa_u = np.random.uniform(0, 1, 1)

    Aaux = a - alfa_u * d_jd_a
    Baux = b - alfa_u * d_jd_b
    dJdAaux, dJdBaux = gradient(x, d, Aaux, Baux, n)

    g = vet_concat(dJdAaux, dJdBaux)
    hl = np.dot(np.transpose(g), dir)

    while (hl < 0):
        alfa_u = 2 * alfa_u
        Aaux = a - alfa_u * d_jd_a;
        Baux = b - alfa_u * d_jd_b;
        dJdAaux, dJdBaux = gradient(x, d, Aaux, Baux, n)
        g = vet_concat(dJdAaux, dJdBaux)
        hl = np.dot(np.transpose(g), dir)

    alfa_m = (alfa_l + alfa_u) / 2
    Aaux = a - alfa_u * d_jd_a;
    Baux = b - alfa_u * d_jd_b;
    dJdAaux, dJdBaux = gradient(x, d, Aaux, Baux, n)

    g = vet_concat(dJdAaux, dJdBaux)
    hl = np.dot(np.transpose(g), dir)

    nit = 0;
    nitmax = np.ceil(np.log((alfa_u - alfa_l) / 1.0e-5))

    while (nit < nitmax and abs(hl) > 1.0e-5):
        nit = nit + 1
        if (hl > 0):
            alfa_u = alfa_m
        else:
            alfa_l = alfa_m
        alfa_m = (alfa_l + alfa_u) / 2;
        Aaux = a - alfa_m * d_jd_a;
        Baux = b - alfa_m * d_jd_b;
        dJdAaux, dJdBaux = gradient(x, d, Aaux, Baux, n)
        g = vet_concat(dJdAaux, dJdBaux)
        hl = np.dot(np.transpose(g), dir)
    alfa = alfa_m
    return alfa


def train(file_name, url, average_error, CONFIGS, RUN):
    f = open(url + RUN.lower() + "/HOG_" + RUN.lower() + "/" + file_name, "r")
    X = np.loadtxt(url + RUN.lower() + "/HOG_" + RUN.lower() + "/" + file_name)
    X = X.reshape(1, len(X))

    if "train_5a" in file_name:
        d = np.array([CONFIGS["MLP_LETTER_Z"]])
    elif "train_53" in file_name:
        d = np.array([CONFIGS["MLP_LETTER_S"]])
    elif "train_58" in file_name:
        d = np.array([CONFIGS["MLP_LETTER_X"]])

    H = CONFIGS["MLP_H"]
    N = np.shape(X)[0]
    ne = CONFIGS["MLP_X_LENGTH"]
    ns = CONFIGS["MLP_NS"]

    A = CONFIGS["MLP_A"]
    B = CONFIGS["MLP_B"]

    ITER_MAX = CONFIGS["MLP_ITER_MAX"]
    ALFA = CONFIGS["MLP_ALFA"]

    # Feedfoward para a saida
    Y = feed_forward(X, A, B, N)
    error = Y - d
    EQM = (1. / N) * ((error * error).sum())

    iter = 0

    vEQM = []
    vEQM.append(EQM)

    while EQM > 1.0e-5 and iter < ITER_MAX:
        iter = iter + 1
        dJdA, dJdB = gradient(X, d, A, B, N)
        ALFA = bis_mlp(X, d, A, B, dJdA, dJdB, N)
        A = A - ALFA * dJdA
        B = B - ALFA * dJdB
        Y = feed_forward(X, A, B, N)
        error = Y - d
        EQM = (1. / N) * ((error * error).sum())
        vEQM.append(EQM)
    Y = feed_forward(X, A, B, N)

    average_error.append(np.average(vEQM))

    CONFIGS = save_train(url + "/config.txt", A, B, average_error, CONFIGS)

    print("Y: {}".format(np.argmax(Y)))
    print("D: {}\n".format(d))

    return CONFIGS, average_error

def save_train(path, A, B, average_error, CONFIGS):
    fr = open(path, "r")

    line = fr.readline()
    lines = []
    while line:
        if "MLP_A : " in line:
            line = line.split(" : ")
            line[1] = ("%s\n" % ';'.join(','.join('%f' % x for x in y) for y in A))
            line = " : ".join("%s" % str(x) for x in line)
        elif "MLP_B : " in line:
            line = line.split(" : ")
            line[1] = ("%s\n" % ';'.join(','.join('%f' % x for x in y) for y in B))
            line = " : ".join("%s" % str(x) for x in line)
        elif "MLP_AVERAGE_ERROR : " in line:
            line = line.split(" : ")
            line[1] = ("%s\n" % ",".join(map(str, average_error)))
            line = " : ".join("%s" % str(x) for x in line)
        lines.append(line)
        line = fr.readline()
    fr.close()

    fw = open(path, "w")
    fw.writelines(lines)
    fw.close()

    CONFIGS["MLP_A"] = A
    CONFIGS["MLP_B"] = B
    CONFIGS["MLP_AVERAGE_ERROR"] = average_error
    return CONFIGS


def test(file_name, url, average_error, CONFIGS, RUN, save):
    f = open(url + RUN.lower() + "/HOG_" + RUN.lower() + "/" + file_name, "r")
    X = np.loadtxt(url + RUN.lower() + "/HOG_" + RUN.lower() + "/" + file_name)
    X = X.reshape(1, len(X))

    if "train_5a" in file_name:
        d = np.array([CONFIGS["MLP_LETTER_Z"]])
    elif "train_53" in file_name:
        d = np.array([CONFIGS["MLP_LETTER_S"]])
    elif "train_58" in file_name:
        d = np.array([CONFIGS["MLP_LETTER_X"]])

    N = np.shape(X)[0]
    A = CONFIGS["MLP_A"]
    B = CONFIGS["MLP_B"]

    # Feedfoward para a saida
    Y = feed_forward(X, A, B, N)
    error = Y - d
    EQM = (1. / N) * ((error * error).sum())

    vEQM = []
    vEQM.append(EQM)

    average_error.append(np.average(vEQM))

    if save:
        CONFIGS = save_test(url + "/config.txt", average_error, CONFIGS)

    print("Y: {}".format(np.argmax(Y)))
    print("D: {}\n".format(d))
    return CONFIGS, average_error


def save_test(path, average_error, CONFIGS):
    fr = open(path, "r")

    line = fr.readline()
    lines = []
    while line:
        if "MLP_AVERAGE_ERROR_TEST : " in line:
            line = line.split(" : ")
            line[1] = ("%s\n" % ",".join(map(str, average_error)))
            line = " : ".join("%s" % str(x) for x in line)
        lines.append(line)
        line = fr.readline()
    fr.close()

    fw = open(path, "w")
    fw.writelines(lines)
    fw.close()

    CONFIGS["MLP_AVERAGE_ERROR_TEST"] = average_error
    return CONFIGS


if __name__ == '__main__':

    url_build = "build/"
    url_test = "testes/"
    url_learning = "treinamento/"

    FOLDER = ""
    RUN = ""

    if "FOLDER" in os.environ:
        FOLDER = os.environ["FOLDER"]

    if "RUN" in os.environ:
        RUN = os.environ["RUN"]

    if RUN:

        # Set path
        os.chdir(url_build)
        for content in os.listdir("."):
            content = content + "/"

            # Define baseline
            url = ""
            if FOLDER:
                url += FOLDER + "/"
                if content != url:
                    continue
            else:
                url += content

            # Define configs
            CONFIGS = configs(url + "/config.txt")

            if RUN == "TREINAMENTO":

                average_error = CONFIGS["MLP_AVERAGE_ERROR"]

                kf = KFold(n_splits=5)

                X = os.listdir(url + RUN.lower() + "/HOG_" + RUN.lower())

                for train, test in kf.split(X):

                    for train_index in train:
                        CONFIGS, average_error = train(X[train_index], url, average_error, CONFIGS, RUN)

                    for test_index in test:
                        CONFIGS, average_error = test(X[test_index], url, average_error, CONFIGS, RUN, False)

            elif RUN == "TESTES":

                average_error = CONFIGS["MLP_AVERAGE_ERROR_TEST"]

                for file_name in os.listdir(url + RUN.lower() + "/HOG_" + RUN.lower()):

                    CONFIGS, average_error = test(file_name, url, average_error, CONFIGS, RUN, True)
