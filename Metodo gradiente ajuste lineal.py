from sympy import *
import math
import timeit

#datos
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

    #Definición de variables
    xi,yi,a,b,t = symbols('xi yi a b t',real = True)
    
    #Función objetivo
    f = (yi - a - (b*xi))**2
    
    fin = True
    cont=0
    
    while fin:

        grad_a=0
        grad_b=0

        da = diff(f,a)
        db = diff(f,b)
        
        #Cálculo de gradiente
        for x1,y1 in data.items():
            grad_a+= da.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})
            grad_b+= db.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})

        #Función objetivo en función del tamaño de paso
        g = f.subs([(a,pa-grad_a*t),(b,pb-grad_b*t)])
        sg=0
        
        for x1,y1 in data.items():
            sg+= g.evalf(subs={xi:x1,yi:y1})
        
        dsg = diff(sg,t)

        #Valor del tamaño de paso óptimo
        tn = solve(dsg,t)[0]

        #Actualización de valores a y b
        pa = pa - tn*grad_a
        pb = pb - tn*grad_b
        
        da = diff(f,a)
        db = diff(f,b)

        grad_a=0
        grad_b=0

        #Se vuelve a calcular el gradiente según la los nuevos valores de a y b
        for x1,y1 in data.items():
            grad_a+= da.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})
            grad_b+= db.evalf(subs={xi:x1,yi:y1,a:pa,b:pb})
            
        #Condición de parada
        if grad_a < error and grad_b < error:
            fin = False

        cont+=1
        print(pa,pb)

    print("El valor óptimo de a es:",pa)
    print("El valor óptimo de b es:",pb)    

    print("El número de iteraciones es:",cont)
        
print("El CPU time es:",timeit.timeit("gradiente()", globals=locals(),number=1),"segundos")




    








        
    
        
    
