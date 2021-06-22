m sympy import *
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

def gradiente(pc=0,pd=0,pe=0,error=0.00001):

    #Definición de variables
    xi,yi,c,d,e,t,wi = symbols('xi yi c d e t wi',real = True)

    #Función objetivo
    f = wi*(yi - c - d*xi - e*(xi**2) )**2

    
    fin = True
    cont=0

    #Pesos
    data_wi_h = [94.60278824052166, 2.522889059845905, 70.24646408646953, 13.984637337532114, 3.9101660386039523, 2.2048364906972586, 15.359901460073434, 53.88650176559752, 2.6185846691324035, 67.93031129900751, 97.71202509264852, 251.98462993744843, 19.782969391221854, 8.472764800965082, 12.922363389104648, 23.489124200446383, 680.3636367288225, 8.581627341909218, 23.12213073645765, 4.2583050071010575]
    
    while fin:

        grad_c=0
        grad_d=0
        grad_e=0
        
        dc = diff(f,c)
        dd = diff(f,d)
        de = diff(f,e)

        #Cálculo de gradiente
        for x1,y1 in data.items():
            grad_c+= dc.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe,wi:data_wi_h[x1-1]})
            grad_d+= dd.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe,wi:data_wi_h[x1-1]})
            grad_e+= de.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe,wi:data_wi_h[x1-1]})

        #Función objetivo en función del tamaño de paso
        g = f.subs([(c,pc-grad_c*t),(d,pd-grad_d*t),(e,pe-grad_e*t)])
        sg=0

        for x1,y1 in data.items():
            sg+= g.evalf(subs={xi:x1,yi:y1,wi:data_wi_h[x1-1]})

        dsg = diff(sg,t)

        #Valor del tamaño de paso óptimo
        tn = solve(dsg,t)[0]

        #Actualización de valores c, d y e
        pc = pc - tn*grad_c
        pd = pd - tn*grad_d
        pe = pe - tn*grad_e
    
        dc = diff(f,c)
        dd = diff(f,d)
        de = diff(f,e)

        grad_c=0
        grad_d=0
        grad_e=0
        
        #Se vuelve a calcular el gradiente según la los nuevos valores de c, d y e
        for x1,y1 in data.items():
            grad_c+= dc.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe,wi:data_wi_h[x1-1]})
            grad_d+= dd.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe,wi:data_wi_h[x1-1]})
            grad_e+= de.evalf(subs={xi:x1,yi:y1,c:pc,d:pd,e:pe,wi:data_wi_h[x1-1]})
        
        #Condición de parada
        if abs(grad_c) < error and abs(grad_d) < error and abs(grad_e) < error:
            fin = False

        cont+=1
        print(pc,pd,pe)

    print("El valor óptimo de c es:",pc)
    print("El valor óptimo de d es:",pd)
    print("El valor óptimo de e es:",pe)
        
    print("El numero de iteraciones es de:",cont)
        

        
    

