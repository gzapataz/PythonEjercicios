##
#Se ingresa un numero entero n
#Devuelve la suma de lo n numeros
#por medio de un ciclo y por medio de la formula n * (n + 1) / 2

while True:
	try:
		n = int(input('Ingrese el valor de n:'))
		break
	except ValueError:
		print('Debe ingresar valores enteros')

suma = 0
suma1 = (n * (n + 1)) / 2
print('Suma con Formula = {}'.format(suma1))

for i in range(n + 1):
	suma = suma + i 
print('Suma sin Formula = {}'.format(suma))
