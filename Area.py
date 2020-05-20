##
# Pide los lados de un area y calcula su area
#
while True:
	try:
		lado1 = float(input('Por favor ingrese los metros lado 1:'))
		lado2 = float(input('Por favor ingrese los metroslado 2:'))
		break
	except ValueError:
		print('Debe ingresar valores numericos')
	
area = lado1 * lado2
print('El area total es:{} Metros Cuadrados'.format(area))
	


