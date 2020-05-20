import random as rd
import numpy as np

def mathArray():
	nparr = np.arange(10)
	nparr1 = np.array([[1, 30, 43, 55, 66], [90, 30,77,98,88]])
	nparr2 = np.random.rand(10,2)
	
	
	#np.random.choice(10)
	
	return nparr, nparr1, nparr2
	

def tuplaSample():
	tl1 = (1,2)

def dictSample():
	dt1 = dict({"nombre":1})

def setSample():
	se1 = set({1, 2, 5, 11})
	se2 = set({5, 7, 11, 29, 31})
	print(se1.intersection(se2))

def main():
	li = list() #forma de crear una lista
	li3 = [] # Crear una lista vacia
	li4 = list(["hola", "Mundo", 12, 13, {2,3}])
	li4 = []
	cad1 = "hola gabriel zapata"
	li5 = cad1.split()
	
	li.append(12)
	li.append(1) 
	li.append(-10)

	l2 = li.copy()
	print(li)
	print(l2)
	
	li3.append("hola")
	li3.append(3.12)
	li3.append(12)
	
	print(li3)
	print(li4)
	for i in li4:
		print(i)
	
	for i in range(0, 20):
		li4.append(rd.randint(0, 200))
	
	li9 = [x ** 2 for x in li4 if x < 30] #Ciclo interno para crear la nueva lista de numeros
	
	print(li4)
	print(li9)
	print(sorted(li5))
	
	li6 = [8, 7, 12, 1, 99]
	print(li6.sort())
	
	setSample()
	
	print('NumpyArange: {} {}'.format(mathArray(), type(mathArray())))
	
main()
