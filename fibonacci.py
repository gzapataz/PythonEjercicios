import numpy as np

global fibonacciSerie

def InitSerie(top):
	global fibonacciSerie
	fibonacciSerie = np.zeros(top)
	print(fibonacciSerie)
	fibonacciSerie[1] = 1
	print(fibonacciSerie)
	return fibonacciSerie
	
def GenerarSerie(top):
	global fibonacciSerie
	fibonacciSerie1 =  InitSerie(top)
	print(fibonacciSerie1)
	print(fibonacciSerie)
	i=2
	while i <= top - 1:
		fibonacciSerie1[i] = fibonacciSerie1[i-1] + fibonacciSerie1[i-2]
		#fibonacciSerie[1] = i
		print(fibonacciSerie1)
		i = i + 1
	print(fibonacciSerie1)


