import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba as nb

@nb.njit
def psi(x_i, x, i, N, h):
    """x - array, i - num of x and function psi_i(x_i), пробная(базисная) функция"""
    if i == 0:
        if x_i >= x[i] and x_i <= x[i+1]:
            return (x[i+1] - x_i) / h
        elif x_i >= x[i+1]:
            return 0
    elif i == N:
        if x_i >= x[i-1] and x_i <= x[N]:
            return (x_i - x[N-1]) / h
        else:
            return 0
    else:        
        if x_i <= x[i-1]:
            return 0
        elif x_i >= x[i-1] and x_i <= x[i]:
            return (x_i - x[i-1]) / h
        elif x_i >= x[i] and x_i <= x[i+1]:
            return (x[i+1] - x_i) / h
        else:
            return 0

@nb.njit(parallel=True)
def M(i,j,x,h,N):
    psi_i = np.zeros((N), dtype = np.float64)
    psi_j = np.zeros((N), dtype = np.float64)
    diff_psi_i = np.zeros((N), dtype = np.float64)
    diff_psi_j = np.zeros((N), dtype = np.float64)
    for n in nb.prange(0,N):
        psi_i[n] = psi(x[n], x, i, N, h)
        psi_j[n] = psi(x[n], x, j, N, h)
    diff_psi_i[1:] = np.diff(psi_i)/h
    diff_psi_j[1:] = np.diff(psi_j)/h
    return np.sum(diff_psi_i*diff_psi_j)*h

@nb.njit(parallel=True)
def K(i,j,x,h,N):
    psi_i = np.zeros((N), dtype = np.float64)
    psi_j = np.zeros((N), dtype = np.float64)
    for n in nb.prange(0,N):
        psi_i[n] = psi(x[n], x, i, N, h)
        psi_j[n] = psi(x[n], x, j, N, h)
    return np.sum(psi_i*psi_j)*h

@nb.njit
def B(i, j, x, h, N):
    x_minus = x[0]   # -L
    x_plus  = x[-1]   # +L

    val_i_plus  = psi(x_plus,  x, i, N, h)
    val_j_plus  = psi(x_plus,  x, j, N, h)
    val_i_minus = psi(x_minus, x, i, N, h)
    val_j_minus = psi(x_minus, x, j, N, h)


    return val_i_plus * val_j_plus + val_i_minus * val_j_minus

@nb.njit(parallel=True)
def Calc_MATRIX(x, h, N):
    M_mtrx = np.zeros((N,N), dtype = np.float64)
    K_mtrx = np.zeros((N,N), dtype = np.float64)
    B_mtrx  = np.zeros((N,N), dtype = np.float64)
    for i in nb.prange(0, N):
        for j in nb.prange(0, N):
            M_mtrx[i,j] = M(i,j,x,h,N)
            K_mtrx[i,j] = K(i,j,x,h,N)
            B_mtrx[i,j] = B(i,j,x,h,N)
    return M_mtrx, K_mtrx, B_mtrx

@nb.njit
def w_n_j(n, j):
    return np.sqrt(n-j+2) - np.sqrt(n-j)

@nb.njit
def sum_right_part(n, Q, N, dt):
    conv = np.zeros(N, dtype=np.complex128)
    for j in range(1, n+1):
        w = w_n_j(n, j)
        conv += (Q[:, j] - Q[:, j-1]) * w
    return conv


@nb.njit(parallel=True)
def Shredinger_calc_ABC(N, T, L, b, Psi_0):
    x = np.linspace(-L, L, N)
    dx = x[1]-x[0]
    dt = (dx**2)
    t = np.linspace(0, dt*T, T)
    h = dx
    
    M_mtrx, K_mtrx, B_mtrx = Calc_MATRIX(x, h, N)
    
    left_part_mtrx = (2/dt) * K_mtrx + 1j*M_mtrx + (2*1j*np.sqrt(b)*np.exp(-1j*np.pi/4)/(np.sqrt(dt*np.pi)))*B_mtrx
    
    right_part_mtrx =  (2/dt) * K_mtrx - 1j*M_mtrx + (2*1j*np.sqrt(b)*np.exp(-1j*np.pi/4)/(np.sqrt(dt*np.pi)))*B_mtrx
    
    right_part_mtrx_caputo = 1j * (2/np.sqrt(dt*np.pi)) * np.sqrt(b) * np.exp(-1j*np.pi/4)
    
    Q_solution = np.zeros((N,T), dtype = np.complex128)
    
    Q_solution[:, 0] = Psi_0[:]     
    
    for i in range(0,T-1):
        
        A = np.dot(right_part_mtrx, Q_solution[:,i])
        
        ABC = np.dot(right_part_mtrx_caputo*B_mtrx , sum_right_part(i, Q_solution, N, T))
        
        Q_solution[:,i+1] = np.linalg.solve(left_part_mtrx, A - ABC) 
    
    Phi = np.zeros((N, N), dtype=np.complex128)
    for i in nb.prange(N):
        for n in nb.prange(N):
            Phi[n, i] = psi(x[n], x, i, N, h)

    Psi = np.zeros((N, T), dtype=np.complex128)
    for m in nb.prange(T):
        Psi[:, m] = np.dot(Phi, Q_solution[:, m])
    
    return Psi, x, t
    
    
@nb.njit
def elbrus_analitic(x, t):
    return (1/(((2*np.pi)**(1/4))*((1+1j*t)**(1/2)))) * np.exp(-(x**2/(4*(1+1j*t))))

L = 25
N = 1000
T = 3000

ksi = 5
x = np.linspace(-L, L, N)
psi_not = elbrus_analitic(x - ksi, 0)/2 + elbrus_analitic(x + ksi, 0)/2 

psi_solv, xx, tt =  Shredinger_calc_ABC(N, T, L, 1, psi_not)

rho_1 = np.zeros((N, T), dtype = np.complex128)

for i in range(0, N):
    for j in range(0, T):
        rho_1[i,j] = elbrus_analitic(xx[i] - ksi, tt[j])/2 + elbrus_analitic(xx[i] + ksi, tt[j])/2 
        
fig, ax = plt.subplots()

# Определяем функцию, которая будет вызываться на каждом кадре анимации
def update(frame):
    # Очищаем предыдущий кадр
    ax.clear()
    # Рисуем новый кадр
    ax.plot(xx, np.abs(psi_solv[:, frame])**2, label='numerical $|\\psi(x,t)|^2$')
    ax.plot(xx, np.abs(rho_1[:, frame])**2, label='analitical $|\\psi(x,t)|^2$')
 
    ax.set_title("Frame {}".format(frame))
    ax.legend()
    ax.set_xlim(xx.min(), xx.max()) 
    ax.set_ylim(0, 0.12)

# Создаем анимацию
ani = animation.FuncAnimation(fig, update, frames=T, interval=0.1)
ani.save("elbrus.gif", writer='pillow', fps=60)
# Показываем анимацию
plt.show()