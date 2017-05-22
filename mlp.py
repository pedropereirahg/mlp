# http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

import numpy as np
from scipy.special import expit
import os
import matplotlib.pyplot as plt


def configs(path):
    configs = {}

    # defaults
    mlp_h = 3
    mlp_letter_z = [0, 1, 0]
    mlp_letter_s = [0, 1, 0]
    mlp_letter_x = [1, 0, 0]

    f = open(path, 'r')
    lines = f.read()
    lines = lines.split("\n")
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
    if configs["MLP_H"]:
        configs["MLP_H"] = int(configs["MLP_H"])
    else:
        f = open(path, 'a')
        f.write("MLP_H : %s\n" % str(mlp_h))
        f.close()

    # Set LETTER_Z default
    if configs["MLP_LETTER_Z"]:
        configs["MLP_LETTER_Z"] = map(int, configs["MLP_LETTER_Z"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_LETTER_Z : %s\n" % ",".join(map(str, mlp_letter_z)))
        f.close()

    # Set LETTER_S default
    if configs["MLP_LETTER_S"]:
        configs["MLP_LETTER_S"] = map(int, configs["MLP_LETTER_S"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_LETTER_S : %s\n" % ",".join(map(str, mlp_letter_s)))
        f.close()

    # Set LETTER_X default
    if configs["MLP_LETTER_X"]:
        configs["MLP_LETTER_X"] = map(int, configs["MLP_LETTER_X"].split(","))
    else:
        f = open(path, 'a')
        f.write("MLP_LETTER_X : %s\n" % ",".join(map(str, mlp_letter_x)))
        f.close()

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


if __name__ == '__main__':

    url_build = "build/"
    url_test = "testes/"
    url_learning = "treinamento/"

    FOLDER = ""
    RUN = ""

    if "FOLDER" in os.environ:
        FOLDER = int(os.environ["FOLDER"])

    if "RUN" in os.environ:
        RUN = os.environ["RUN"]

    if RUN:

        # Set path
        os.chdir(url_build)
        for content in os.listdir("."):

            # Define baseline
            url = url_build
            if FOLDER:
                url += FOLDER + "/"
                if content != url:
                    break
            else:
                url += content

            # Define configs
            CONFIGS = configs(url + RUN.lower() + "/config.txt")

            # N = np.shape(X)[0]
            # ne = CONFIGS["MLP_X_LEN"]
            # ns = np.shape(d)[1]

            A = np.random.rand(H, (ne + 1))  # FIXAR
            B = np.random.rand(ns, (H + 1))  # FIXAR

            if RUN == "TREINAMENTO": # CROSS VALIDATION

                for file_name in os.listdir(url + RUN.lower() + "/HOG_" + RUN.lower()):
                    X = np.loadtxt(file_name)
                    X = X.reshape(1, len(X))

                    if "train_5a" in file_name:
                        d = np.array([CONFIGS["MLP_LETTER_Z"]])
                    elif "train_53" in file_name:
                        d = np.array([CONFIGS["MLP_LETTER_S"]])
                    elif "train_58" in file_name:
                        d = np.array([CONFIGS["MLP_LETTER_X"]])

                    H = CONFIGS["MLP_LETTER_Z"]
                    N = np.shape(X)[0]
                    ne = CONFIGS["MLP_X_LENGTH"]
                    ns = np.shape(d)[1]

                    A = np.random.rand(H, (ne + 1))  # FIXAR
                    B = np.random.rand(ns, (H + 1))  # FIXAR

                    # Feedfoward para a saida
                    Y = feed_forward(X, A, B, N)
                    error = Y - d
                    EQM = (1. / N) * ((error * error).sum())

                    i = 0
                    alfa = 1

                    vEQM = []
                    vEQM.append(EQM)

                    while EQM > 1.0e-5 and i < 10000:
                        i = i + 1
                        dJdA, dJdB = gradient(X, d, A, B, N)
                        alfa = bis_mlp(X, d, A, B, dJdA, dJdB, N)
                        A = A - alfa * dJdA
                        B = B - alfa * dJdB
                        Y = feed_forward(X, A, B, N)
                        error = Y - d
                        EQM = (1. / N) * ((error * error).sum())
                        vEQM.append(EQM)
                    Y = feed_forward(X, A, B, N)

                    print("Y: {}".format(np.argmax(Y)))
                    print("D: {}\n".format(d))

            # elif RUN == "TESTES":
