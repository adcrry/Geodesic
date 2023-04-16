import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import *

#g = [[g_11 , g_12], [g_21, g_22]]

x, y = symbols('x y')

#Normalization of the last two components for X=[x_1,x_2,dx_1/dt,dx_2/dt]
def normalize_speed(X): 
    return [X[0],X[1],X[2]/(np.sqrt(X[2]**2 + X[3]**2)),X[3]/(np.sqrt(X[2]**2 + X[3]**2))]

#Normalization of u=[x1,x2,x3]
def normalize(u):
    n = sqrt(u[0]**2+u[1]**2+u[2]**2)
    return [u[0]/n,u[1]/n,u[2]/n]

#u=[x1,x2,x3] and v=[y1,y2,y3]
def dot(u,v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

#Differentiate a vector-valued function
def diffv(f, s):
    return [diff(f[0], s), diff(f[1], s) ,diff(f[2], s)]

#Cross product of u=[x1,x2,x3] and v=[y1,y2,y3]
def cross(u,v):
    return [u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]

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

#Get the gaussian curvature
def gauss(f):
    global x
    global y
    f_x = diffv(f, x)
    f_y = diffv(f, y)
    N = cross(f_x,f_y)
    N = normalize(N)
    N_x = diffv(N, x)
    N_y = diffv(N, y)
    g = metric(f)[0]

    det_g = g[0][0]*g[1][1] - g[0][1]**2
    det_h = dot(N_x,f_x)*dot(N_y,f_y)-dot(N_x,f_y)**2


    return simplify(det_h/det_g)

def F_gamma(X,gamma):
    global x
    global y

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


def initial_speed_2p(X_0,X_1,f,f_l):
    global x
    global y

    x_0 = [f_l[0](X_0[0], X_0[1]), f_l[1](X_0[0], X_0[1]), f_l[2](X_0[0], X_0[1])] #coordinates in R^3
    x_1 = [f_l[0](X_1[0], X_1[1]), f_l[1](X_1[0], X_1[1]), f_l[2](X_1[0], X_1[1])]

    v = np.array(x_1) - np.array(x_0)

    dx1 = diffv(f, x)
    dx2 = diffv(f, y)

    dx1_l = [lambdify([x,y], dx1[0]), lambdify([x,y], dx1[1]), lambdify([x,y], dx1[2])]
    dx2_l = [lambdify([x,y], dx2[0]), lambdify([x,y], dx2[1]), lambdify([x,y], dx2[2])]

    dx1_0 = [dx1_l[0](X_0[0], X_0[1]), dx1_l[1](X_0[0], X_0[1]), dx1_l[2](X_0[0], X_0[1])]
    dx2_0 = [dx2_l[0](X_0[0], X_0[1]), dx2_l[1](X_0[0], X_0[1]), dx2_l[2](X_0[0], X_0[1])]

    N = cross(dx1_0, dx2_0)
    N = normalize(N)

    v = v - dot(N, v)*np.array(N) # projection of v on the tangent space at X_0

    v = [float(v[0]),float(v[1]),float(v[2])]

    A = np.array([[np.dot(dx1_0,dx1_0),np.dot(dx1_0,dx2_0)],[np.dot(dx1_0,dx2_0),np.dot(dx2_0,dx2_0)]])
    b = np.array([np.dot(v,dx1_0), np.dot(v,dx2_0)])

    z = np.linalg.solve(A,b)

    return z

#Parameterization of the manifold as an embedded submanifold of R^3
#WARNING : Remember that a parametrization f need to be an immersion i.e. df never vanishes, if not it is possible that g^-1 diverge !
f = [x,y,sqrt(1-x**2-y**2)] #f : U -> R^3 with U an open subset of R^2

f_l = [lambdify([x,y], f[0]), lambdify([x,y], f[1]), lambdify([x,y], f[2])] #Converting f to a "numerical" function and not a formal one

D_f = 0.7 #[-D_f,D_f]^2 is a square domain center at 0 where f is defined
X_0 = [0,0,7,3] #initial condition as [x_1,x_2,dx_1/dt,dx_2/dt] where x_i is the ith coordinate of the curve defined by f^-1 on the manifold


#Here we are trying to get a geodesic between p_0 and p_1
p_0 = [0,0]
p_1 = [0.3,0.3]
z = initial_speed_2p(p_0,p_1,f,f_l)
X_0 = [p_0[0],p_0[1],z[0],z[1]]



X_0 = normalize_speed(X_0) #normalization of the initial velocity vector

g, inv_g = metric(f)
gamma = christoffel(g, inv_g)

F = lambda t, X : F_gamma(X,gamma)

t_eval = np.arange(0, D_f, 0.01) #if x_1=x_2=0 be aware if not and in general to not leave the domain of f
sol = solve_ivp(F, [0, D_f], X_0, t_eval=t_eval)

print("K =", gauss(f)) #printing the gaussian curvature of the surface

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

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6, cmap='viridis', edgecolor='none') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim3d(-D_f,D_f)
ax.set_ylim3d(-D_f,D_f)
ax.set_zlim3d(-D_f,D_f)

p_0 = [f_l[0](p_0[0],p_0[1]),f_l[1](p_0[0],p_0[1]),f_l[2](p_0[0],p_0[1])]
p_1 = [f_l[0](p_1[0],p_1[1]),f_l[1](p_1[0],p_1[1]),f_l[2](p_1[0],p_1[1])]

ax.scatter(p_0[0], p_0[1], p_0[2], c='green', marker='*', s=50)
ax.scatter(p_1[0], p_1[1], p_1[2], c='blue', marker='*', s=50)

plt.show()

