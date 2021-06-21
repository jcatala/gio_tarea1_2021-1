from sympy import *
import math
import timeit

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


def gradiente(pa=1,pb=1,error=0.00001):
    xi,yi,a,b,t = symbols('xi yi a b t',real = True)
    f = (yi - a - (b*xi))**2
    fin = True
    cont=0
    
    while fin:

        grad_a=0
        grad_b=0
        
        da = diff(f,a)
        db = diff(f,b)
        
        for x1,y1 in data.items():
            grad_a+= da.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})
            grad_b+= db.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})

        g = f.subs([(a,pa-grad_a*t),(b,pb-grad_b*t)])
        sg=0
        
        for x1,y1 in data.items():
            sg+= g.evalf(subs={xi:x1,yi:y1})
        
        dsg = diff(sg,t)
        tn = solve(dsg,t)[0]
        
        pa = pa - tn*grad_a
        pb = pb - tn*grad_b
        
        da = diff(f,a)
        db = diff(f,b)

        grad_a=0
        grad_b=0
        
        for x1,y1 in data.items():
            grad_a+= da.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})
            grad_b+= db.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})
        
        if grad_a < error and grad_b < error:
            fin = False

        cont+=1
        print(pa,pb)

        print(grad_a,grad_b)

    print("El numero de iteraciones es de:",cont)
        
print("El CPU time es de:",timeit.timeit("gradiente()", globals=locals(),number=1),"segundos")


    








        
    
        
    
