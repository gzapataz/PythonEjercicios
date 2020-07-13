##
# Funcion Head, muestra los 10 primeros renglones de un archivo
# Recibe de parametro el nombre del archivo y muestra sus primeras 10 lineas
#
import sys
from pathlib import Path

class ArgumentRequestedError(Exception):
    pass

# Head del archivo numLines primeras lineas del archivo
def printFile(file, numlines):
    try:
        f = open(file, mode="r")
        for i in range(0, numlines):
            print(f.readline())
            i = i + 1
        f.close()
    except FileExistsError:
        print('Error: Archivo no existe')
    except Exception as e:
        print('Error General:{}'.format(str(e)))

# Tail del archivo numLines primeras lineas del archivo
def tailFile(file, offset):
    contador = 0
    bytesSize = []
    f = open(file, mode="r")
    for x in f:
        bytesSize.append(len(x))
        contador = contador + 1
    print(f'Contador de Lineas:{contador} total {sum(bytesSize)}')
    print(f'offsets {bytesSize[offset:]}')
    f.seek(Path(file).stat().st_size - sum(bytesSize[offset:]), 0)
    for x1 in f:
        print(x1)




def main(fileArg, numlines):
    if numlines >= 0:
        printFile(fileArg, numlines)
    else:
        tailFile(fileArg, numlines)

if __name__ == '__main__':
    print(f"Arguments count: {len(sys.argv)}")
    numelines = 10
    try:
        if len(sys.argv) < 2:
            raise ArgumentRequestedError("File Error")
        else:
            numelines = sys.argv[2]
        main(sys.argv[1], int(numelines))
    except ArgumentRequestedError as e:
        print(f'Debe indicar el nombre del archivo: {e}')