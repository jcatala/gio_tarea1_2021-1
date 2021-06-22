# Tarea número 1 de GIO 2021-1


## Summary

Tarea la cual tiene el objetivo, realizar la implementación de 2 métodos de descenso (Gradiente y Newton), utilizando `exact line search` para conseguir t. Esto con el objetivo de encontrar la mejor aproximación al problema de mínimos cuadrados.


# Uso

## Newton


Para utilizar el método de newton, correr el programa con python3:

```python
python3 newton.py
```

Si se desea cambiar el punto de partida, editar la siguiente linea a los puntos deseados:

```python
x0 = (0,0,0)
```

Al correr el programa, podemos ver la información de cada iteración, en conjunto de su respectivo paso 

```python
➜  repo git:(main) ✗ python3 newton.py 
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
Starting point: (0.000000, 0.000000)
[ 1 ] Current point (1.246769, 0.477082), t value: 1.000000
[ 2 ] Current point (1.246769, 0.477082), t value: 0.934926

Final point ( 1.246769, 0.477082 ), took 0.533718 [secs] 
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
Starting point: (0.000000, 0.000000, 0.000000)
[ 1 ] Current point (2.922830, -0.027857, 0.024930), t value: 1.000000
[ 2 ] Current point (2.922830, -0.027857, 0.024930), t value: 0.405502

Final point ( 2.922830, -0.027857, 0.024930 ), took 0.907353 [secs] 
Ended

```



## Gradiente

Para utilizar el método de `gradiente`, existen 2 archivos:

* `Metodo gradiente ajuste cuadratico.py`
* `Metodo gradiente ajuste lineal.py`

En donde para editar el valor del punto de partida, es necesario editar la llamada a cada función

```python
# Gradiente para aproximación cuadrática
def gradiente(pc=10,pd=10,pe=10,error=0.00001):

# Gradiente para aproximación lineal
def gradiente(pa=0,pb=0,error=0.00001):
```

Para correr el programa

```python
➜  repo git:(main) ✗ python3 Metodo\ gradiente\ ajuste\ cuadratico.py
...
...
2.92282995505469 -0.0278570947452212 0.0249302902241222
2.92282995407547 -0.0278570945660492 0.0249302902257091
2.92282995407536 -0.0278570945665792 0.0249302902169201
2.92282995310287 -0.0278570943886435 0.0249302902184552
2.92282995310276 -0.0278570943891675 0.0249302902097677
2.92282995214717 -0.0278570942143228 0.0249302902112813
2.92282995214706 -0.0278570942148379 0.0249302902027395
2.92282995120477 -0.0278570940424263 0.0249302902042224
2.92282995120466 -0.0278570940429337 0.0249302901958091
2.92282995027303 -0.0278570938724744 0.0249302901972543
2.92282995027293 -0.0278570938729748 0.0249302901889571
2.92282994934870 -0.0278570937038718 0.0249302901903556
2.92282994934859 -0.0278570937043662 0.0249302901821595
2.92282994843449 -0.0278570935371185 0.0249302901835200
2.92282994843439 -0.0278570935376061 0.0249302901754365
2.92282994754255 -0.0278570933744278 0.0249302901767960
2.92282994754245 -0.0278570933749055 0.0249302901688772
El valor óptimo de c es: 2.92282994754245
El valor óptimo de d es: -0.0278570933749055
El valor óptimo de e es: 0.0249302901688772
El numero de iteraciones es de: 2233
El CPU time es: 524.4351283979995 segundos

```