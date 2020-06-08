##
#Evalua una expresion, la convierte a postfija y devuelve el resultado
#

import tokenizeString as tk
import InfixToPostfix as ipos

def evaluatePostfix(postfix):
	valuesList = []
	for token in postfix:
		if token.lstrip('-').isdigit(): #Evalua si es digito teniendo en cuenta si el primero es un -
			valuesList.append(int(token))
		else:
			right = valuesList.pop()
			left = valuesList.pop()
			if token == '^':
				token = '**'
			newValue = eval(str(left) + token + str(right))
			valuesList.append(int(newValue))
	return valuesList.pop()
		
		

def main():
	tokenList = []
	postfix = [] # Va guardando la cadena en postfijo
	evaluacion = []
	
	stToken = input('Expresion Matematica Valida:')
	tokenList = tk.tokenizeStr(stToken)
	print('La cadena en tokens es: {}'.format(tokenList))
	postfix = ipos.convertPostfix(tokenList)
	print('La cadena en postifijo es: {}'.format(postfix))
	evaluacion = evaluatePostfix(postfix)
	print('El total es: {}'.format(evaluacion))

if __name__ == '__main__':
	main()
