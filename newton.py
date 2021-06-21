#!/usr/bin/venv python3

import sympy as sp
import numpy as np
import time

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

data_xi = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
data_yi = [2.910348,2.429320,3.034274,3.564630,3.995516,4.404796,3.766659,4.225016,5.367586,5.065610,5.572047,6.271426,6.567767,7.082164,7.829593,8.632454,9.657083,10.106290,11.533470,12.736908]
data_wi = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


sp.init_printing() # math as latex



def calculate_gradient(fun, symbols=[] ):
    return sp.derive_by_array(fun, symbols)

def calculate_hessian(fun, symbols = []):
    return sp.derive_by_array(calculate_gradient(fun, symbols),symbols)




def do_newton_h(iter = 10):
    return -1


def get_paso_newton_g(f_hess_inv, f_grad,a0,b0, symbols):
    values = f_hess_inv(data_xi, data_yi, data_wi)
    #delta_newton = (-1) * values * f_grad(data_xi,data_yi,data_wi)
    #sp.pprint(values)
    #sp.pprint(f_grad(data_xi,data_yi,data_wi))
    temp = (-1) * sp.Matrix(values) * (sp.Matrix(f_grad(data_xi,data_yi,data_wi)))
    #sp.pprint(temp)
    #sp.pprint(delta_newton)
    #x = delta_newton[0][0].subs([(symbols[0],a0),(symbols[1],b0)])
    #y = delta_newton[1][0].subs([(symbols[0],a0),(symbols[1],b0)])
    x = temp[0].subs([(symbols[0],a0),(symbols[1],b0)])
    y = temp[1].subs([(symbols[0],a0),(symbols[1],b0)])
    return [x,y]

def get_lambdacuad(f_grad, paso_newton, a0, b0, symbols):
    jacob_ev = f_grad(data_xi,data_yi,data_wi)
    #sp.pprint(jacob_ev)
    #sp.pprint(paso_newton)
    lambda_pow2  = jacob_ev[0].subs([(symbols[0], a0), (symbols[1], b0)]) * (-1) * paso_newton[0]
    lambda_pow2 += jacob_ev[1].subs([(symbols[0], a0), (symbols[1], b0)]) * (-1) * paso_newton[1]
    return lambda_pow2

def exact_line_g(f_ag, f_grad , a0, b0, symbols):
    t = symbols[-1]
    a = symbols[0]
    b = symbols[1]
    # First, evaluate the grad
    grad_values = f_grad(data_xi,data_yi,data_wi)
    grad_evaluated = [grad_values[0].subs([(a,a0),(b,b0)]),\
         grad_values[1].subs([(a,a0),(b,b0)])]
    grad_evaluated = t * sp.Matrix(grad_evaluated)
    f_arg = sp.Matrix([a0,b0]) + grad_evaluated # Create the f argument to minimize
    # Evaluate the result
    result = f_ag(data_xi, data_yi, data_wi).subs([(a,f_arg[0]), (b, f_arg[1] )])
    # derivate and solve for t == 0
    result_dif = sp.diff(result, t)
    result_dif = sp.solve(result_dif)
    return result_dif

def do_newton_g(iter = 10):
    lambdaCuad = 10
    err = 0.004
    #wi, xi, yi = sp.symbols('wi xi yi', cls=sp.Idx)

    wi, xi, yi, a, b, c, t, n,i  = sp.symbols('wi xi yi a b c t n i')

    #b,a,x,y,w,i = sym.symbols('b a x y w i')
    #f_ab = sp.summation( sp.Indexed('wi',i) * (sp.Indexed('yi',i) - (b * sp.Indexed('xi',i) + a))** 2, (i,0,19))
    #f = sp.lambdify([xi,yi,wi], err)

    #f_ab = wi * (yi - a - b*xi)**2
    #f_ab = sp.summation( ww[n] * (( yy[n] - a  - b*xx[n]))**2, (n, 0, 19) )
    f_ab = sp.Sum(sp.Indexed('wi',i)  * (sp.Indexed('yi',i) - a - b*sp.Indexed('xi',i))**2, (i,0,19) )
    f = sp.lambdify([xi, yi, wi], f_ab)
    sp.pprint(f_ab)
    #print(f(data_xi, data_yi, data_wi).subs([(a,0),(b,0)]) )

    a0 = 0
    b0 = 0
    startTime = time.time()
    iteramount = 0 
    f_ab = f_ab
    f_grad = calculate_gradient(f_ab, symbols=[a,b]).doit()
    f_grad_value = sp.lambdify([xi, yi, wi] , f_grad)
    
    f_hess = calculate_hessian(f_ab, symbols=[a,b])
    f_hess_inv = sp.Matrix(f_hess).inv()
    f_hess_inv_value = sp.lambdify([xi, yi, wi], f_hess_inv)

    while lambdaCuad / 2 >= err:
        paso_newton = get_paso_newton_g(f_hess_inv_value, f_grad_value,a0,b0,[a,b])
        lambdaCuad = get_lambdacuad(f_grad_value, paso_newton, a0, b0, [a,b])
        paso = exact_line_g(f, f_grad_value, a0, b0, symbols=[a,b,t])[0]
        #a0 = a0 + paso * paso_newton[0]
        #b0 = b0 + paso * paso_newton[1]
        grad_numbers = f_grad_value(data_xi,data_yi,data_wi)
        a0 = a0 + paso * grad_numbers[0].subs([(a,a0),(b,b0)])
        b0 = b0 + paso * grad_numbers[1].subs([(a,a0),(b,b0)])
        iteramount = iteramount + 1
        print("[ %d ] Current point (%f, %f), t value: %f" % (iteramount, a0, b0, paso), end="\n")

    print("")
    print("Final point ( %f, %f ), took %f [secs] " % (a0, b0, time.time() - startTime))
    print("Ended")

def get_paso_newton_h(f_hess_inv, f_grad,c0,d0,e0,symbols):
    c = symbols[0]
    d = symbols[1]
    e = symbols[2]
    #values = f_hess_inv(data_xi, data_yi, data_wi)
    #delta_newton = (-1) * values * f_grad(data_xi,data_yi,data_wi)
    #sp.pprint(values)
    #sp.pprint(f_grad(data_xi,data_yi,data_wi))
    delta_newton = (-1) * f_hess_inv * (sp.Matrix(f_grad(data_xi,data_yi,data_wi)))
    x = delta_newton[0].subs([(c,c0),(d,d0),(e,e0)])
    y = delta_newton[1].subs([(c,c0),(d,d0),(e,e0)])
    z = delta_newton[2].subs([(c,c0),(d,d0),(e,e0)])
    return [x,y,z]

def get_lambdacuad_h(f_grad, paso_newton, c0,d0,e0, symbols):
    c = symbols[0]
    d = symbols[1]
    e = symbols[2]
    t = symbols[3]
    jacob_ev = f_grad(data_xi,data_yi,data_wi)
    #print("---")
    #sp.pprint(jacob_ev)
    #sp.pprint(paso_newton)
    lambda_pow2  =  jacob_ev[0].subs([(c, c0), (d, d0), (e, e0)]) * (-1) * paso_newton[0]
    lambda_pow2  += jacob_ev[1].subs([(c, c0), (d, d0), (e, e0)]) * (-1) * paso_newton[1]
    lambda_pow2  += jacob_ev[2].subs([(c, c0), (d, d0), (e, e0)]) * (-1) * paso_newton[2]
    return lambda_pow2



def exact_line_h(f_h, f_grad , c0, d0 ,e0, symbols):
    t = symbols[-1]
    c = symbols[0]
    d = symbols[1]
    e = symbols[2]
    # First, evaluate the grad
    grad_values = f_grad(data_xi,data_yi,data_wi)
    #sp.pprint(grad_values)
    grad_evaluated = [grad_values[0].subs([(c,c0),(d,d0),(e,e0)]),\
        grad_values[1].subs([(c,c0),(d,d0),(e,e0)]),\
        grad_values[2].subs([(c,c0),(d,d0),(e,e0)])]
    grad_evaluated = t * sp.Matrix(grad_evaluated)
    f_arg = sp.Matrix([c0,d0,e0]) + grad_evaluated # Create the f argument to minimize
    # Evaluate the result
    result = f_h(data_xi, data_yi, data_wi).subs([(c,f_arg[0]),(d,f_arg[1]), (e,f_arg[2]) ])
    # derivate and solve for t == 0
    result_dif = sp.diff(result, t)
    result_dif = sp.solve(result_dif)
    return result_dif

def do_newton_h():
    err = 0.004

    wi, xi, yi, c, d, e, t, n, i = sp.symbols('wi xi yi c d e t n i')

    f_cde = sp.Sum(sp.Indexed('wi',i) * ( sp.Indexed('yi',i) - c - d*sp.Indexed('xi',i) - e*(sp.Indexed('xi',i))**2  )**2, (i,0,19) ) 
    f_val = sp.lambdify([xi, yi, wi], f_cde)

    f_grad = calculate_gradient(f_cde, symbols=[c,d,e])
    sp.pprint(f_grad)
    f_grad = f_grad.doit()
    f_grad_value = sp.lambdify([xi, yi, wi], f_grad)
    

    f_hess = calculate_hessian(f_cde, symbols=[c,d,e]).doit()
    f_hess_lamb = sp.lambdify([xi, yi, wi], f_hess)
    f_hess_lamb = f_hess_lamb(data_xi, data_yi, data_wi)
    sp.pprint(f_hess_lamb)
    f_hess_inv = sp.Matrix(f_hess_lamb).inv()

    #sp.pprint(f_hess_inv)
    #f_hess_inv_value = sp.lambdify([xi, yi, wi],f_hess_inv)

    # Set initial values
    c0,d0,e0 = (2,0,0)
    lambdaCuad = 100
    startTime = time.time()
    iteramount = 0
    while lambdaCuad/2 >= err:
        paso_newton = get_paso_newton_h(f_hess_inv, f_grad_value, c0, d0, e0, [c, d, e])
        lambdaCuad = get_lambdacuad_h(f_grad_value, paso_newton, c0, d0, e0, [c,d,e,t])
        # ---
        paso = exact_line_h(f_val, f_grad_value, c0, d0, e0, symbols=[c, d, e, t])[0]
        grad_numbers = f_grad_value(data_xi,data_yi,data_wi)
        c0 = c0 + paso * grad_numbers[0].subs([ (c,c0), (d,d0), (e,e0)   ])
        d0 = d0 + paso * grad_numbers[1].subs([ (c,c0), (d,d0), (e,e0)   ])
        e0 = e0 + paso * grad_numbers[2].subs([ (c,c0), (d,d0), (e,e0)   ])
        iteramount = iteramount + 1
        print("[ %d ] Current point (%f, %f, %f), t value: %f" % (iteramount, c0, d0, e0, paso), end="\n")

    print("")
    print("Final point ( %f, %f, %f ), took %f [secs] " % (c0, d0, e0, time.time() - startTime))
    print("Ended")
    sp.pprint(f_cde)

if __name__ == "__main__":


    #do_newton_g(iter = 10)
    do_newton_h()
