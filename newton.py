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


data_wi_g = data_wi
data_wi_h = data_wi
data_wi_g = [0.5989004542062513, 9.473867615628091, 5.082181985659734, 4.204686993718601, 5.35085892855351, 7.9205584660845, 1.6906802612959755, 1.5743640126241567, 50.53483481532306, 1.158923031302284, 1.210955982095339, 2.066633056754798, 1.275232930127079, 1.3603011579583153, 2.813016965621227, 12.78887888913651, 14.937556686238732, 20.348448518917394, 0.7398388302904214, 0.28299968414484483]
data_wi_h = [94.60278824052166, 2.522889059845905, 70.24646408646953, 13.984637337532114, 3.9101660386039523, 2.2048364906972586, 15.359901460073434, 53.88650176559752, 2.6185846691324035, 67.93031129900751, 97.71202509264852, 251.98462993744843, 19.782969391221854, 8.472764800965082, 12.922363389104648, 23.489124200446383, 680.3636367288225, 8.581627341909218, 23.12213073645765, 4.2583050071010575]


sp.init_printing() # math as latex



def calculate_gradient(fun, symbols=[] ):
    return sp.derive_by_array(fun, symbols)

def calculate_hessian(fun, symbols = []):
    return sp.derive_by_array(calculate_gradient(fun, symbols),symbols)





def get_paso_newton_g(f_hess_inv, f_grad,a0,b0, symbols):
    temp = (-1) * sp.Matrix(f_hess_inv) * (sp.Matrix(f_grad(data_xi,data_yi,data_wi_g)))

    x = temp[0].subs([(symbols[0],a0),(symbols[1],b0)])
    y = temp[1].subs([(symbols[0],a0),(symbols[1],b0)])
    return [x,y]

def get_lambdacuad_g(f_grad, paso_newton, a0, b0, symbols):
    jacob_ev = f_grad(data_xi,data_yi,data_wi_g)

    lambda_pow2  = jacob_ev[0].subs([(symbols[0], a0), (symbols[1], b0)]) * (-1) * paso_newton[0]
    lambda_pow2 += jacob_ev[1].subs([(symbols[0], a0), (symbols[1], b0)]) * (-1) * paso_newton[1]
    return lambda_pow2

def exact_line_g(f_ag, paso_newton, a0, b0, symbols):
    t = symbols[-1]
    a = symbols[0]
    b = symbols[1]
    # First, evaluate the grad
    p_new = [paso_newton[0], paso_newton[1]]
    grad_evaluated =   t * sp.Matrix(p_new)

    f_arg = sp.Matrix([a0,b0]) + grad_evaluated # Create the f argument to minimize
    # Evaluate the result
    result = f_ag(data_xi, data_yi, data_wi_g).subs([(a,f_arg[0]), (b, f_arg[1] )])
    # derivate and solve for t == 0
    result_dif = sp.diff(result, t)
    result_dif = sp.solve(result_dif)
    return result_dif

    

def do_newton_g(x0):
    lambdaCuad = 10
    err = 0.004
    #wi, xi, yi = sp.symbols('wi xi yi', cls=sp.Idx)

    wi, xi, yi, a, b, c, t, n,i  = sp.symbols('wi xi yi a b c t n i')

    f_ab = sp.Sum(sp.Indexed('wi',i)  * (sp.Indexed('yi',i) - a - b*sp.Indexed('xi',i))**2, (i,0,19) )
    f = sp.lambdify([xi, yi, wi], f_ab)
    sp.pprint(f_ab)
    sp.pprint("Initializing ... ")
    f_ab = f_ab
    f_grad = calculate_gradient(f_ab, symbols=[a,b]).doit()
    f_grad_value = sp.lambdify([xi, yi, wi] , f_grad)
    

    f_hess = calculate_hessian(f_ab, symbols=[a,b]).doit()
    f_hess_lamb = sp.lambdify([xi, yi, wi], f_hess)
    f_hess_lamb = f_hess_lamb(data_xi, data_yi, data_wi_g)
    f_hess_inv = sp.Matrix(f_hess_lamb).inv()
    f_hess_inv_value = f_hess_inv

    a0,b0 = x0
    startTime = time.time()
    iteramount = 0 
    sp.pprint("Starting ...")
    print("Starting point: (%f, %f)" % (a0, b0))
    while lambdaCuad / 2 >= err:
        paso_newton = get_paso_newton_g(f_hess_inv_value, f_grad_value,a0,b0,[a,b])
        lambdaCuad = get_lambdacuad_g(f_grad_value, paso_newton, a0, b0, [a,b])
        paso = exact_line_g(f, paso_newton, a0, b0, symbols=[a,b,t])[0]
        a0 = a0 + paso * paso_newton[0]
        b0 = b0 + paso * paso_newton[1]
        iteramount = iteramount + 1
        print("[ %d ] Current point (%f, %f), t value: %f" % (iteramount, a0, b0, paso), end="\n")

    print("")
    print("Final point ( %f, %f ), took %f [secs] " % (a0, b0, time.time() - startTime))
    print("Ended")

def get_paso_newton_h(f_hess_inv, f_grad,c0,d0,e0,symbols):
    c = symbols[0]
    d = symbols[1]
    e = symbols[2]

    delta_newton = (-1) * f_hess_inv * (sp.Matrix(f_grad(data_xi,data_yi,data_wi_h)))
    x = delta_newton[0].subs([(c,c0),(d,d0),(e,e0)])
    y = delta_newton[1].subs([(c,c0),(d,d0),(e,e0)])
    z = delta_newton[2].subs([(c,c0),(d,d0),(e,e0)])
    return [x,y,z]

def get_lambdacuad_h(f_grad, paso_newton, c0,d0,e0, symbols):
    c = symbols[0]
    d = symbols[1]
    e = symbols[2]
    t = symbols[3]
    jacob_ev = f_grad(data_xi,data_yi,data_wi_h)

    lambda_pow2  =  jacob_ev[0].subs([(c, c0), (d, d0), (e, e0)]) * (-1) * paso_newton[0]
    lambda_pow2  += jacob_ev[1].subs([(c, c0), (d, d0), (e, e0)]) * (-1) * paso_newton[1]
    lambda_pow2  += jacob_ev[2].subs([(c, c0), (d, d0), (e, e0)]) * (-1) * paso_newton[2]
    return lambda_pow2



def exact_line_h(f_h, paso_newton , c0, d0 ,e0, symbols):
    t = symbols[3]
    c = symbols[0]
    d = symbols[1]
    e = symbols[2]
    # First, evaluate the grad
    p_new = [paso_newton[0], paso_newton[1] , paso_newton[2]]
    p_new =  t * sp.Matrix(p_new)
    f_arg = sp.Matrix([c0,d0,e0]) + p_new # Create the f argument to minimize
    # Evaluate the result
    result = f_h(data_xi, data_yi, data_wi_h).subs([(c,f_arg[0]),(d,f_arg[1]), (e,f_arg[2]) ])
    # derivate and solve for t == 0
    result_dif = sp.diff(result, t)
    result_dif = sp.solve(result_dif)
    #   sp.pprint(result_dif)
    return result_dif

def do_newton_h(x0):

    wi, xi, yi, c, d, e, t, n, i = sp.symbols('wi xi yi c d e t n i')

    f_cde = sp.Sum(sp.Indexed('wi',i) * ( sp.Indexed('yi',i) - c - d*sp.Indexed('xi',i) - e*(sp.Indexed('xi',i))**2  )**2, (i,0,19) ) 
    f_val = sp.lambdify([xi, yi, wi], f_cde)

    sp.pprint(f_cde)
    f_grad = calculate_gradient(f_cde, symbols=[c,d,e])
    #sp.pprint(f_grad)
    f_grad = f_grad.doit()
    f_grad_value = sp.lambdify([xi, yi, wi], f_grad)
    

    f_hess = calculate_hessian(f_cde, symbols=[c,d,e]).doit()
    f_hess_lamb = sp.lambdify([xi, yi, wi], f_hess)
    f_hess_lamb = f_hess_lamb(data_xi, data_yi, data_wi_h)
    #sp.pprint(f_hess_lamb)
    f_hess_inv = sp.Matrix(f_hess_lamb).inv()

    # Set initial values
    sp.pprint("Starting")
    c0,d0,e0 = x0
    err = 0.004
    lambdaCuad = 100
    startTime = time.time()
    iteramount = 0
    print("Starting point: (%f, %f, %f)" % (c0, d0, e0))
    while lambdaCuad/2 >= err:
        # Get paso, lambda cuad, and t
        paso_newton = get_paso_newton_h(f_hess_inv, f_grad_value, c0, d0, e0, [c, d, e])
        lambdaCuad = get_lambdacuad_h(f_grad_value, paso_newton, c0, d0, e0, [c,d,e,t])
        paso = exact_line_h(f_val, paso_newton, c0, d0, e0, symbols=[c, d, e, t])[0]
        
        grad_numbers = f_grad_value(data_xi,data_yi,data_wi_h)
        c0 = c0 + paso * paso_newton[0]
        d0 = d0 + paso * paso_newton[1]
        e0 = e0 + paso * paso_newton[2]

        iteramount = iteramount + 1
        print("[ %d ] Current point (%f, %f, %f), t value: %f" % (iteramount, c0, d0, e0, paso), end="\n")

    print("")
    print("Final point ( %f, %f, %f ), took %f [secs] " % (c0, d0, e0, time.time() - startTime))
    print("Ended")

if __name__ == "__main__":

    x0 = (0,0,0)
    do_newton_g( x0[:2])
    do_newton_h(x0)
