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


def gradiente(pa=0,pb=0,error=0.00001):

    #Definición de variables
    xi,yi,a,b,t,wi = symbols('xi yi a b t wi',real = True)
    
    #Función objetivo
    f = wi * (yi - a - (b*xi))**2
    
    fin = True
    cont=0
    
    #Pesos
    data_wi_g = [0.5989004542062513, 9.473867615628091, 5.082181985659734, 4.204686993718601, 5.35085892855351, 7.9205584660845, 1.6906802612959755, 1.5743640126241567, 50.53483481532306, 1.158923031302284, 1.210955982095339, 2.066633056754798, 1.275232930127079, 1.3603011579583153, 2.813016965621227, 12.78887888913651, 14.937556686238732, 20.348448518917394, 0.7398388302904214, 0.28299968414484483] 
    
    while fin:

        grad_a=0
        grad_b=0

        da = diff(f,a)
        db = diff(f,b)
        
        #Cálculo de gradiente
        for x1,y1 in data.items():
            
            grad_a+= da.evalf(subs={xi:x1,yi:y1,a:pa,b:pb,wi:data_wi_g[x1-1]})
            
            grad_b+= db.evalf(subs={xi:x1,yi:y1,a:pa,b:pb,wi:data_wi_g[x1-1]})

        #Función objetivo en función del tamaño de paso
        g = f.subs([(a,pa-grad_a*t),(b,pb-grad_b*t)])
        sg=0
        
        for x1,y1 in data.items():
            sg+= g.evalf(subs={xi:x1,yi:y1,wi:data_wi_g[x1-1]})
        
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
            grad_a+= da.evalf(subs={xi:x1,yi:y1,a:pa,b:pb,wi:data_wi_g[x1-1]})
            grad_b+= db.evalf(subs={xi:x1,yi:y1,a:pa,b:pb,wi:data_wi_g[x1-1]})
            
        #Condición de parada
        if grad_a < error and grad_b < error:
            fin = False

        cont+=1
        print(pa,pb)

    print("El valor óptimo de a es:",pa)
    print("El valor óptimo de b es:",pb)    

    print("El número de iteraciones es:",cont)
        
print("El CPU time es:",timeit.timeit("gradiente()", globals=locals(),number=1),"segundos")








        
    
        
    
