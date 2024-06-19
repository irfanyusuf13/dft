import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 2, 4, 8, 16, 32, 64, 128], dtype=complex)
N = len(x)

dft = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        dft[k, n] = np.exp(-2j * np.pi * k * n / N)

spektral = np.dot(dft, x)

dft_inv = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        dft_inv[k, n] = np.exp(2j * np.pi * k * n / N) / N

x_semula = np.dot(dft_inv, spektral)

plt.figure(figsize=(8, 6))
plt.stem(np.arange(N), np.abs(spektral), basefmt=" ")
plt.title('Vektor Spektral')
plt.xlabel('Frekuensi')
plt.ylabel('Magnitudo')
plt.grid(True)
plt.show()


print("Matriks Transformasi DFT\n")
print(dft)
print("Vektor Spektral\n")
print(spektral)
print("Matriks Invers DFT\n")
print(dft_inv)
print("Vektor Semula")
print(np.round(x_semula.real)) 
