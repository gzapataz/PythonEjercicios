##
# Calculo de retorno de dinero por devolver botellas vacias

while True:
	try:
		cont1litro = float(input('Numero de contenedores de menos de un litro?'))
		contmaslitro = float(input('Numero de contenedores de mas de un litro?'))
		break
	except ValueError:
		print('Debe ingresar valores numericos')

montopequenos = 0.10
montomayores = 0.25

montodevolver = montomayores * contmaslitro + montopequenos * cont1litro

print('El monto a devolver es: {0:.2f} $'.format(montodevolver))

