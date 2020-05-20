##
# Tokeniza una cadena matematica valida 

def getNum(strToken):
	i = 0
	while i < len(strToken):
		if strToken[i].isdigit() == False and i != 0:
			break
		i = i + 1
	return strToken[0:i], i

# Tokeniza un string con formula matematica
def tokenizeStr(stToken):
	i = 0
	stListTokenized = []
	while i < len(stToken): # Itera mientras haya caracteres en la cadena
		if stToken[i].isdigit(): #Evalua si es digito toma el numero
			elem, i = getNum(stToken[i:])
		elif stToken[i] in ['-']: # Evalua si es digito precedido de un menos para validar si es operador unario
			if stToken[i-1].lstrip().isdigit() and i > 0: # revisa que el anterior sea digito y que ademas no sea el primer elemento vara evaluar menos binario
				elem = stToken[i]
				stToken = stToken[i+1:]
				i = 0
			else: # Separa el numero con un - unario
				elem, i = getNum(stToken[i:])
		elif stToken[i] in ['*', '/', '^', '(', ')', '+']: # Si es operador binario lo asigna y continua
			elem = stToken[i]
			stToken = stToken[i+1:]
			i = 0
		else: # Si no es ninguno de las anteriores continua
			elem = ''
			stToken = stToken[i+1:]
			i = 0
		if elem != '':
			stListTokenized.append(elem)
	return stListTokenized

def main():
	tokenList = []
	stToken = input('Ingrese la expresion Matematica Valida:')
	tokenList = tokenizeStr(stToken)
	print('La cadena tokenizada es: {}'.format(tokenList))

if __name__ == '__main__':
	main()
