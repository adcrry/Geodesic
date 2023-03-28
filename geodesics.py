import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import *

#g = [[g_11 , g_12], [g_21, g_22]]

x, y = symbols('x y')

#Normalization of the last two components for X=[x_1,x_2,dx_1/dt,dx_2/dt]
def normalize_speed(X): 
    return [X[0],X[1],X[2]/(np.sqrt(X[2]**2 + X[3]**2)),X[3]/(np.sqrt(X[2]**2 + X[3]**2))]

#u=[x1,x2,x3] and v=[y1,y2,y3]
def dot(u,v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

#differantiate a vector-valued function
def diffv(f, s):
    return [diff(f[0], s), diff(f[1], s) ,diff(f[2], s)]

#Get the metric and its inverse from the parametrization
def metric(f):
    global x
    global y
    f_x = diffv(f, x)
    f_y = diffv(f, y)   
    g = [[simplify(dot(f_x,f_x)),simplify(dot(f_x,f_y))],[simplify(dot(f_x,f_y)),simplify(dot(f_y,f_y))]]
    det = g[0][0]*g[1][1] - g[0][1]**2
    inv_g = [[simplify(g[1][1]/det), simplify(- g[0][1]/det)],[simplify(- g[0][1]/det), simplify(g[0][0]/det)]] 
    return [g, inv_g]

#Get the christoffel symbols from the metric and its inverse
def christoffel(g, inv_g):
    global x
    global y

    gamma = [[],[]] # gamma[0]=[gamma^x_xx, gamma^x_xy, gamma^x_yy] and gamma[1]=[gamma^y_xx, gamma^y_xy, gamma^y_yy]

    for k in range(0,2):
        for l in range(0,3):
            i = x
            j = x
            if l >= 1:
                j = y
            if l == 2:
                i = y
            
            c = lambda a : 0 if a == x else 1
            cv = lambda a : x if a == 0 else y

            s = 0
            for m in range(0,2):
                s += inv_g[k][m]*(diff(g[c(j)][m], i)+diff(g[c(i)][m], j) - diff(g[c(i)][c(j)], cv(m)))

            gamma[k].append(simplify(0.5*s))

    return gamma

def F_gamma(X,gamma):
    global x
    global Y

    Y=[]

    for k in range(2,4):
        s = 0
        for i in range(0,2):
            for j in range(0,2):
                if i == 0:
                    s += X[i+2]*X[j+2]*gamma[k-2][j].subs([(x, X[0]), (y, X[1])]).evalf()
                else:
                    s += X[i+2]*X[j+2]*gamma[k-2][j+1].subs([(x, X[0]), (y, X[1])]).evalf()
        Y.append(-s)
    return [X[2],X[3],Y[0],Y[1]]

#Parameterization of the manifold as an embedded submanifold of R^3
#WARNING : Remember that a parametrization f need to be an immersion i.e. df never vanishes, if not it is possible that g^-1 diverge !
f = [x,-y,x*y] #f : U -> R^3 with U an open subset of R^2
f_l = [lambdify([x,y], f[0]), lambdify([x,y], f[1]), lambdify([x,y], f[2])] #Converting f to a "numerical" function and not a formal one

D_f = 1 #[-D_f,D_f]^2 is a square domain center at 0 where f is defined
X_0 = [0,0,0.5,-1] #initial condition as [x_1,x_2,dx_1/dt,dx_2/dt] where x_i is the ith coordinate of the curve defined by f^-1 on the manifold

X_0 = normalize_speed(X_0) #normalization of the initial velocity vector

g, inv_g = metric(f)
gamma = christoffel(g, inv_g)

F = lambda t, X : F_gamma(X,gamma)

t_eval = np.arange(0, D_f, 0.1) #if x_1=x_2=0 be aware if not and in general to not leave the domain of f
sol = solve_ivp(F, [0, D_f], X_0, t_eval=t_eval)


geodesic = np.array([f_l[0](sol.y[0],sol.y[1]),  f_l[1](sol.y[0],sol.y[1]), f_l[2](sol.y[0],sol.y[1])])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])


X = geodesic[0]
Y = geodesic[1]
Z = geodesic[2]

ax.plot(X, Y, Z, '-r', linewidth = 3)


U,V = np.meshgrid(np.linspace(-D_f, D_f, 30),np.linspace(-D_f, D_f, 30))

X = f_l[0](U,V)
Y = f_l[1](U,V)
Z = f_l[2](U,V)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none') #for now, the surface need to be defined like this f=[x,y,g(x,y)]

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim3d(-D_f,D_f)
ax.set_ylim3d(-D_f,D_f)
ax.set_zlim3d(-D_f,D_f)

plt.show()

