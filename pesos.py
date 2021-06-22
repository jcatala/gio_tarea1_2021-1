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

'''
  19                               
 ___                               
 ╲                                 
  ╲                         2      
  ╱   (-a - b⋅xi[i] + yi[i]) ⋅wi[i]
 ╱                                 
 ‾‾‾                               
i = 0                              
Initializing ... 
Starting ...
Starting point: (100.000000, 100.000000)
[ 1 ] Current point (1.131908, 0.486261), t value: 1.000000
[ 2 ] Current point (1.131908, 0.486261), t value: 1.064057

Final point ( 1.131908, 0.486261 ), took 0.358165 [secs] 
Ended
  19                                          
 ____                                         
 ╲                                            
  ╲                                           
   ╲                                   2      
   ╱  ⎛                      2        ⎞       
  ╱   ⎝-c - d⋅xi[i] - e⋅xi[i]  + yi[i]⎠ ⋅wi[i]
 ╱                                            
 ‾‾‾‾                                         
i = 0                                         
Starting
Starting point: (1000.000000, 1000.000000, 1000.000000)
[ 1 ] Current point (3.016370, -0.027683, 0.024474), t value: 1.000000
[ 2 ] Current point (3.016370, -0.027683, 0.024474), t value: 0.990407

Final point ( 3.016370, -0.027683, 0.024474 ), took 0.496142 [secs] 

'''
def pesos_g():
    # considerando ajuste de curva con pesos unitarios
    x,y = sp.symbols('x y')
    f = 1.131908 + 0.486261*x
    f_l = sp.lambdify([x], f)
    weights = []
    for k in range(20):
        exp     = data_yi[k]
        aprox   = f_l(data_xi[k])
        err = abs(exp - aprox)
        wi = 1/(err**2)
        weights.append(wi)
        #sp.pprint("Experimental: %f, teorico: %f, err: %f, wi: %f" % (exp, aprox, err, wi))
    sp.pprint("Weights for g(a,b):")
    sp.pprint(weights)

def pesos_h():
    x,y,z = sp.symbols('x y z')
    f = 3.016370 + -0.027683*x+ 0.024474*(x**2)
    sp.pprint(f)
    f_l = sp.lambdify([x], f)
    weights = []
    for k in range(20):
        exp     = data_yi[k]
        aprox   = f_l(data_xi[k])
        err     = abs(exp - aprox)
        wi = 1/(err**2)
        weights.append(wi) 
        #sp.pprint("Experimental: %f, teorico: %f, err: %f, wi: %f" % (exp, aprox, err, wi))
    sp.pprint("Weights for h(c,d,e):")
    sp.pprint(weights)

if __name__ == "__main__":
    pesos_g()
    pesos_h()