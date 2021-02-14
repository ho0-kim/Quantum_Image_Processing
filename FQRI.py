# Ref.)
# https://github.com/Shedka/citiesatnight/blob/5f17f1b323740cea77623ad038a8e713c25849db/frqi.py
# https://arxiv.org/pdf/1812.11042.pdf
# https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute
from qiskit.extensions import UnitaryGate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import datasets, transforms

def genGraycode(nbit):
    graycode = ['0', '1']

    for _ in range(nbit-1):
        mirrored = reversed(graycode)
        graycode = ['0'+g for g in graycode]
        graycode = graycode + ['1'+m for m in mirrored]
    
    return graycode

def MCRY(theta, n_controlbit):
    t = 2**(n_controlbit + 1)
    mat = np.identity(t)
    # higher qubit indices are more significant (little endian convention)
    #      ┌───┐
    # q_0: ┤ X ├
    #      └─┬─┘
    # q_1: ──■──
    #        │
    # q_2: ──■──
    # mat[t-2, t-2] = np.cos(np.pi/4)
    # mat[t-2, t-1] = -np.sin(np.pi/4)
    # mat[t-1, t-2] = np.sin(np.pi/4)
    # mat[t-1, t-1] = np.cos(np.pi/4)

    # q_0: ──■──
    #        │
    # q_1: ──■──
    #      ┌─┴─┐
    # q_2: ┤ X ├
    #      └───┘
    mat[t//2-1, t//2-1] = np.cos(theta/2)
    mat[t//2-1, t-1] = -np.sin(theta/2)
    mat[t-1, t//2-1] = np.sin(theta/2)
    mat[t-1, t-1] = np.cos(theta/2)

    return UnitaryGate(mat, label=f'multi-controlled ry gate {theta}')

if __name__ == '__main__':
    # X_train = datasets.MNIST(root='./data', train=True, download=True,
    #                      transform=transforms.Compose([transforms.ToTensor()]))

    img_folder = "C:/Users/thesi/Downloads/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST - JPG - training/0/"
    img_path = img_folder + "1.jpg"

    img = cv2.imread(img_path, 0)

    """ image reszie from 28x28 to 32x32 """
    img_size = 16
    img = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)

    """ Image Visualization """
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',img)
    # k = cv2.waitKey(0)
    # if k == 27:         # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'): # wait for 's' key to save and exit
    #     cv2.imwrite('messigray.png',img)
    #     cv2.destroyAllWindows()

    """ 2d to 1d """
    print(img.shape)
    img = img.reshape(img_size*img_size)
    print(img.shape)

    """ Nomralization (0~1 == sin(0)~sin(pi/2)) """
    img = img / 255.0

    """ calculate theta """
    img = np.arcsin(img)

    """ Create Multi-controlled RY Gate """
    k = int(np.ceil(np.log2(img_size)))

    """ Design circuit """
    qc = QuantumCircuit(2*k+1, 2*k+1)

    qc.h(range(2*k))

    graycode = genGraycode(2*k)

    qc.barrier()
    for i, e in enumerate(img):
        if i == 0:
            qc.x(range(0, 2*k))
        else:
            for j in range(2*k):
                if graycode[i-1][j] != graycode[i][j]:
                    qc.x(j)
                    break
        mcry = MCRY(2 * img[i], 2*k)
        qc.append(mcry, range(2*k+1))
    qc.barrier()

    qc.measure(range(2*k+1), range(2*k+1))

    """ run simulator """
    provider = IBMQ.load_account()
    #backend = provider.get_backend('ibmq_qasm_simulator')
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, shots=8000, seed_simulator=12345, backend_options={"fusion_enable":True})
    #job = execute(qc, backend=backend, shots=8192)
    result = job.result()
    count = result.get_counts()
    plot_histogram(count)

    #### decode
    for i in range(len(x_train[img_num])):
        try:
            genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'01']/numOfShots)])
        except KeyError:
            genimg = np.append(genimg,[0.0])

    # inverse nomalization
    genimg *= 32.0 * 255.0
    x_train = np.sin(x_train)
    x_train *= 255.0

    # convert type
    genimg = genimg.astype('int')

    # back to 2-dimentional data
    genimg = genimg.reshape((28,28))

    plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
    plt.savefig('gen_'+str(img_num)+'.png')
    plt.show()