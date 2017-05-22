# http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

import numpy as np
from scipy.special import expit
import os
import matplotlib.pyplot as plt

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

    url_dataset = "dataset/"
    url_test = "testes/"
    url_learning = "treinamento/"
    url_sample = "sample/"

    pixels_per_celL = 8
    cells_per_block = 1
    orientations = 9
    run = ",,"

    if "PIXELS_PER_CELL" in os.environ:
        pixels_per_celL = int(os.environ["PIXELS_PER_CELL"])

    if "CELLS_PER_BLOCK" in os.environ:
        cells_per_block = int(os.environ["CELLS_PER_BLOCK"])

    if "ORIENTATIONS" in os.environ:
        orientations = int(os.environ["ORIENTATIONS"])

    if "RUN" in os.environ:
        run = os.environ["RUN"]

    run = run.split(",")

    url_result = "build/PCP-" + repr(pixels_per_celL) + "-CPB-" + repr(cells_per_block) + "/"

    for layer in run:

        # Set path
        os.chdir(url_result + layer.lower() + "/HOG_" + layer.lower())

        # if layer == "TREINAMENTO":
        #     X = ""  # CROSS VALIDATION
        #
        # elif layer == "TESTES":
        #     X = ""

        for file_name in os.listdir("."):
            f = open(file_name, 'r')
            X = np.loadtxt(file_name)
            X = X.reshape(1, len(X))

            if "train_5a" in file_name:
                d = np.array([[0, 1, 0]])  # Z
            elif "train_53" in file_name:
                d = np.array([[0, 0, 1]])  # S
            elif "train_58" in file_name:
                d = np.array([[1, 0, 0]])  # X

            aux = np.shape(X)
            h = 3
            N = aux[0]
            ne = aux[1]
            aux = np.shape(d)
            ns = aux[1]

            A = np.random.rand(h, (ne + 1))  # FIXAR
            B = np.random.rand(ns, (h + 1))  # FIXAR

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
