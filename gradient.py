#!/usr/bin/venv python3
import time


# Less weight to a bigger error
# https://www.cimat.mx/~joaquin/cursos/mat251/clases/clase11.pdf

data = {
1   :2.910348,
2   :2.429320,
3   :3.034274,
4   :3.564630,
5   :3.995516,
6   :4.404796,
7   :3.766659,
8   :4.225016,
9   :5.367586,
10  :5.065610,
11  :5.572047,
12  :6.271426,
13  :6.567767,
14  :7.082164,
15  :7.829593,
16  :8.632454,
17  :9.657083,
18  :10.106290,
19  :11.533470,
20  :12.736908
}

# R(X) function which is using the g(x) function to get the error
def Rxy_g(a, b, weight=1):
    
    s = 0
    for xi, yi in data.items():
        # Change the weight after (we're gonna use a dictionary of weights in order to get the weight with the "key" variable)
        w = 1
        s += w * ( (yi - a - (b*xi) )**2 )
    return s

# R(x) function which has been derivated by a
def R_grad_a_g(a,b,weight=1):

    s = 0
    for xi, yi in data.items():
        w = 1
        s += w * ( -2*yi + 2*b*xi + 2*a)
    return s


def R_grad_b_g(a, b, weight=1):
    s = 0

    for xi, yi in data.items():
        w = 1
        s += w * (-2*yi*xi + 2*a*xi + 2*b*(xi**2) )
    return s

# R(x) function which is made by using the h(x) function
def Rxy_h(c, d, e, weight = 1):
    s = 0
    for xi, yi in data.items():
        w = 1
        s += w * ( c**2 + 2*c*d*xi - 2*c*yi + 2*e*c*(xi**2) + (d**2)*(xi**2) - 2*d*yi*xi + 2*e*d*(xi**3) + (yi**2) - 2*e*yi*(xi**2) + (e**2)*(xi**4))
    return s

def R_grad_c_h(c, d, e, weight = 1):
    s = 0
    for xi, yi in data.items():
        w = 1
        s += w * ( 2*c + 2*d*xi - 2*yi + 2*e*(xi**2) )
    return s

    
def R_grad_e_h(c, d, e, weight = 1):
    s = 0
    for xi, yi in data.items():
        w = 1
        s += w * ( 2*c*xi + 2*d*(xi**2) - 2*yi*xi + 2*e*(xi**3) )
    return s

def R_grad_d_h(c, d, e, weight = 1):
    s = 0
    for xi, yi in data.items():
        w = 1
        s += w * ( 2*c*(xi**2) + 2*d*(xi**3) - 2*yi*(xi**2) + 2*e*(xi**4) )
    return s



def BacktrackingLineSearch_g(x0,y0):
    t = 1
    x = x0
    y = y0
    beta = 0.6
    alpha = 0.3
    c = 1e-4
    backtrackingIteration = 1
    while Rxy_g( x + t * (-R_grad_a_g(x,y)) , y + t * (-R_grad_b_g(x,y)) ) >\
         (Rxy_g( x,y )) + (alpha * t  * ( -R_grad_a_g( x, y )*R_grad_a_g(x , y ) + -R_grad_b_g(x , y) * R_grad_b_g(x , y ))):
        #print("new t: {}".format(t))
        #time.sleep(0.01)
        print("[ {} ] Backtracking Iteration, new t: {}".format(backtrackingIteration, t), end = "\r" )
        backtrackingIteration += 1
        t *= beta
    print("[ {} ] Final t of backtracking t: {}".format(backtrackingIteration, t), end = "\n" )
    return t

def GradientDescent_g(n):
    
    initialPoint = (20,12.736908)
    #initialPoint = (1,2.910348)
    
    print("Using the initial point P0: ( %d, %d )" % (initialPoint[0], initialPoint[1]) )
    error = 100

    x0 = initialPoint[0]
    y0 = initialPoint[1]
    next_x0 = x0
    next_y0 = y0
    startTime = time.time()
    current_iter = 0
    
    while error > 0.004:
        stepSize = BacktrackingLineSearch_g(x0, y0) # Armijo condition
        anterior = Rxy_g(x0, y0)
        next_x0 = x0 + stepSize * ( - R_grad_a_g(x0, y0) )
        next_y0 = y0 + stepSize * ( - R_grad_b_g(x0, y0) )
        
        actual = Rxy_g(next_x0, next_y0)
        error = abs(actual - anterior)

        x0 = next_x0
        y0 = next_y0
        print("""[ %d ] New point is: (%f, %f), with error of: %f""" % (current_iter, x0, y0, error))
        current_iter += 1
        if(current_iter == n): break
    print("Final point is: (%f, %f) with %f iterations" % (x0, y0, current_iter))
    return startTime, current_iter



def BacktrackingLineSearch_h(c0, d0, e0):
    t = 1
    c = c0
    d = d0
    e = e0

    beta = 0.6
    alpha = 0.3
    c = 1e-4
    backtrackingIteration = 1
    while Rxy_h( c + t * (-R_grad_c_h(c,d,e)) , d + t * (-R_grad_d_h(c,d,e)), e + t * (-R_grad_e_h(c,d,e))  ) >\
         (Rxy_h( c,d,e )) + (alpha * t  * ( -R_grad_c_h( c,d,e )*R_grad_c_h(c,d,e ) + -R_grad_d_h( c,d,e )*R_grad_d_h(c,d,e ) + -R_grad_e_h( c,d,e )*R_grad_e_h(c,d,e ) ) ):
        #print("new t: {}".format(t))
        time.sleep(0.01)
        print("[ {} ] Backtracking Iteration, new t: {}".format(backtrackingIteration, t), end = "\r" )
        backtrackingIteration += 1
        t *= beta
    print("[ {} ] Final t of backtracking t: {}".format(backtrackingIteration, t), end = "\n" )
    return t



def GradientDescent_h(n):
    
    initialPoint = (10,100,20)
    #initialPoint = (1,2.910348)
    
    print("Using the initial point P0: ( %d, %d, %d )" % (initialPoint[0], initialPoint[1], initialPoint[2]) )
    error = 100

    c0 = initialPoint[0]
    d0 = initialPoint[1]
    e0 = initialPoint[2]

    next_c0 = c0
    next_d0 = d0
    next_e0 = e0

    startTime = time.time()
    current_iter = 0
    
    while error > 0.004:
        stepSize = BacktrackingLineSearch_h(c0, d0, e0) # Armijo condition
        anterior = Rxy_h(c0, d0, e0)
        next_c0 = c0 + stepSize * ( - R_grad_c_h(c0, d0, e0) )
        next_d0 = d0 + stepSize * ( - R_grad_d_h(c0, d0, e0) )
        next_e0 = e0 + stepSize * ( - R_grad_e_h(c0, d0, e0) )
        
        actual = Rxy_h(next_c0, next_d0, next_e0 )
        error = abs(actual - anterior)

        c0 = next_c0
        d0 = next_d0
        e0 = next_e0

        print("""[ %d ] New point is: (%f, %f, %f), with error of: %f""" % (current_iter, c0, d0, e0, error))
        current_iter += 1
        if(current_iter == n): break
    print("Final point is: (%f, %f, %f) with %f iterations" % (c0, d0, e0, current_iter))
    return startTime, current_iter


def solve_linear(n = 10):
    GradientDescent_g(n)


def solve_cuadratic(n = 10):
    GradientDescent_h(n)

if __name__ == "__main__":
    #data_sorted = sorted(data.items(), key=lambda x: x[1], reverse=True)
    print(data)
    print(Rxy_g(1,1,0))
    n = input("Numbers of iterations to run ? (Default: 10)\n> ")
    #solve_linear(int(n))
    solve_cuadratic(10)

