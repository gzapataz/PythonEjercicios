'''
Programa de practica para revisar los conceptos de:
Variables Globales y Locales
Retorno de multiples valores
Paso de variables por nombre y por posicion
'''

x = 11


def suma():
	global x
	print("global X: ", x)
	global y # Crea una variable local para todo el progrma
	y = x 
	print(y)
	
def sumaLocal():
	x = 33
	print("Local:", x)
	
def paramTest(a, b, c, d, sumando):
	return a + sumando, b + sumando, c + sumando, d + sumando
	
	
if __name__ == "__main__":
	suma()
	sumaLocal()
	suma()
	a, b, c, d = paramTest(2, 3, 4, 5, 10)
	print(a,b,c,d)
	
	a, b, c, d = paramTest(sumando = 10, d = 5, c = 4, a = 2, b = 3)
	print(a,b,c,d, y)
	

