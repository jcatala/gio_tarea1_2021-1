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

