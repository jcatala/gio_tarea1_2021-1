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

def gradiente(pc=0,pd=0,pe=0,error=0.00001):
    xi,yi,c,d,e,t = symbols('xi yi c d e t',real = True)
    f = (yi - c - d*xi - e*(xi**2) )**2
    fin = True
    cont=0

    while fin:

        grad_c=0
        grad_d=0
        grad_e=0
        
        dc = diff(f,c)
        dd = diff(f,d)
        de = diff(f,e)

        for x1,y1 in data.items():
            grad_c+= dc.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe})
            grad_d+= dd.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe})
            grad_e+= de.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe})

        g = f.subs([(c,pc-grad_c*t),(d,pd-grad_d*t),(e,pe-grad_e*t)])
        sg=0

        for x1,y1 in data.items():
            sg+= g.evalf(subs={xi:x1,yi:y1})

        dsg = diff(sg,t)
        tn = solve(dsg,t)[0]

        pc = pc - tn*grad_c
        pd = pd - tn*grad_d
        pe = pe - tn*grad_e
    
        dc = diff(f,c)
        dd = diff(f,d)
        de = diff(f,e)

        grad_c=0
        grad_d=0
        grad_e=0

        for x1,y1 in data.items():
            grad_c+= dc.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe})
            grad_d+= dd.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe})
            grad_e+= de.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe})
    
        if abs(grad_c) < error and abs(grad_d) < error and abs(grad_e) < error:
            fin = False

        cont+=1
        print(pc,pd,pe)
        
    print("El numero de iteraciones es de:",cont)
        
print("El CPU time es de:",timeit.timeit("gradiente()", globals=locals(),number=1),"segundos")


        

        
    

