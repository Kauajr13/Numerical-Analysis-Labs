import curses
import numpy as np

def menu(stdscr, op): # Criando Menu
    stdscr.clear()
    menu = ["CalculoDeterminante", "SistemaTriangularInferior", "SistemaTriangularSuperior", "DecomposicaoLU", "Cholesky", "GaussCompacto", "GaussJordan", "Jacobi", "GaussSeidel", "MatrizInversa"]
    titulo = "\n\t\t\t[*] Menu de Metodos Numericos[*]\n"
    
    # Exibir titulo
    stdscr.addstr(0, 0, titulo, curses.A_BOLD)
    
    # Exibir opcoes do menu
    for idx, row in enumerate(menu):
        x = 0
        y = idx + 3
        if idx == op:
            stdscr.addstr(y, x, f"->  {row}  <-")
        else:
            stdscr.addstr(y, x, f"   {row}")
    x = 5
    y += 2
    stdscr.addstr(y+1, x, "[?] Use as setas cima e baixo do teclado para navegar")  # Ajuda
    stdscr.addstr(y+2, x, "[?] Pressione Q para sair")
    stdscr.addstr(y+3, x, "[!] Sempre digite numeros ao entrar")

    stdscr.refresh()

def get_input(stdscr, prompt): # Codigo para usar imput no stdscr
    stdscr.clear()
    stdscr.addstr(0, 0, prompt)
    stdscr.refresh()

    curses.echo()
    user_input = stdscr.getstr().decode('utf-8')
    curses.noecho()

    return user_input

def main(stdscr): # Funcao para mexer no menu e selecionar a funcao desejada
    curses.curs_set(0)  # Esconder cursor
    op = 0
    
    menu(stdscr, op)
    
    while True:
        tecla = stdscr.getch()
        
        if tecla == curses.KEY_UP and op > 0:
            op -= 1
        elif tecla == curses.KEY_DOWN and op < 9:
            op += 1
        elif tecla == ord('q'):
            break
        elif tecla == curses.KEY_DOWN and op == 9:
            op = 0
        elif tecla == curses.KEY_UP and op == 0:
            op = 9
        elif tecla == curses.KEY_ENTER or tecla in [10, 13]: # Tecla enter pressionada
            
            
            stdscr.clear()

            stdscr.refresh()

            curses.curs_set(1)

            if op == 0:
                CalculoDeterminante(stdscr)
            elif op == 1:
                SistemaTriangularInferior(stdscr)
            elif op == 2:
                SistemaTriangularSuperior(stdscr)
            elif op == 3:
                DecomposicaoLU(stdscr)
            elif op == 4:
                Cholesky(stdscr)
            elif op == 5:
                GaussCompacto(stdscr)
            elif op == 6:
                GaussJordan(stdscr)
            elif op == 7:
                Jacobi(stdscr)
            elif op == 8:
                GaussSeidel(stdscr)
            elif op == 9:
                MatrizInversa(stdscr)
            
        
        menu(stdscr, op)
        curses.curs_set(0)


def CalculoDeterminante(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr,"Digite a ordem da matriz: "))
    
    # Inicializa a matriz vazia
    matriz = []
    
    # Solicita os elementos da matriz
    print(f"Digite os elementos da matriz {ordem}x{ordem}:")
    stdscr.getch()
    for i in range(ordem):
        linha = []
        for j in range(ordem):
            elemento = float(get_input(stdscr, f"Elemento [{i+1},{j+1}]: "))
            linha.append(elemento)
        matriz.append(linha)
    
    # Converte a lista de listas em um array numpy
    matriz_np = np.array(matriz)
    
    # Calcula o determinante
    determinante = np.linalg.det(matriz_np)
    
    # Converte o determinante para inteiro
    determinante_int = int(round(determinante))
    
    print("O resultado é " + str(determinante_int)) 
    curses.curs_set(0)
    stdscr.getch()

def SistemaTriangularInferior(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz vazia
    matriz = []
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem} (triangular inferior):\n")
    stdscr.getch()
    for i in range(ordem):
        linha = []
        for j in range(ordem):
            if j <= i:
                elemento = float(get_input(stdscr, f"Elemento [{i+1},{j+1}]: "))
            else:
                elemento = 0.0
            linha.append(elemento)
        matriz.append(linha)
    
    # Solicita o vetor dos termos independentes
    vetor = []
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr,f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" +  f"Elemento [{i+1}]: "))
        vetor.append(elemento)
    
    # Converte a lista de listas em um array numpy
    matriz_np = np.array(matriz)
    vetor_np = np.array(vetor)
    
    # Inicializa o vetor solução
    solucao = np.zeros(ordem)
    
    # Resolve o sistema triangular inferior
    for i in range(ordem):
        soma = 0
        for j in range(i):
            soma += matriz_np[i, j] * solucao[j]
        solucao[i] = (vetor_np[i] - soma) / matriz_np[i, i]
    
    # Imprime a solução
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {solucao[i]:.4f}\n")
    
    curses.curs_set(0)
    stdscr.getch()
def SistemaTriangularSuperior(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz vazia
    matriz = []
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem} (triangular superior):\n")
    for i in range(ordem):
        linha = []
        for j in range(ordem):
            if j >= i:
                elemento = float(get_input(stdscr, f"Digite os elementos da matriz {ordem}x{ordem} (triangular superior):\n" + f"Elemento [{i+1},{j+1}]: "))
            else:
                elemento = 0.0
            linha.append(elemento)
        matriz.append(linha)
    
    # Solicita o vetor dos termos independentes
    vetor = []
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr,f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" +  f"Elemento [{i+1}]: "))
        vetor.append(elemento)
    
    # Converte a lista de listas em um array numpy
    matriz_np = np.array(matriz)
    vetor_np = np.array(vetor)
    
    # Inicializa o vetor solução
    solucao = np.zeros(ordem)
    
    # Resolve o sistema triangular superior
    for i in range(ordem-1, -1, -1):
        soma = 0
        for j in range(i+1, ordem):
            soma += matriz_np[i, j] * solucao[j]
        solucao[i] = (vetor_np[i] - soma) / matriz_np[i, i]
    
    # Imprime a solução
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {solucao[i]:.4f}\n")
    
    curses.curs_set(0)
    stdscr.getch()
def DecomposicaoLU(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz vazia
    matriz = []
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem}:\n")
    for i in range(ordem):
        linha = []
        for j in range(ordem):
            elemento = float(get_input(stdscr, f"Digite os elementos da matriz {ordem}x{ordem}:\n" + f"Elemento [{i+1},{j+1}]: "))
            linha.append(elemento)
        matriz.append(linha)
    
    # Solicita o vetor dos termos independentes
    vetor = []
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr,f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" +  f"Elemento [{i+1}]: "))
        vetor.append(elemento)
    
    # Converte a lista de listas em um array numpy
    matriz_np = np.array(matriz)
    vetor_np = np.array(vetor)
    
    # Decomposição LU
    P, L, U = lu_decomposicao(matriz_np)
    
    # Resolve o sistema Ax = b usando a decomposição LU
    y = np.linalg.solve(L, np.dot(P, vetor_np))  # Ly = Pb
    x = np.linalg.solve(U, y)  # Ux = y
    
    # Imprime a solução
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {x[i]:.4f}\n")
    
    curses.curs_set(0)
    stdscr.getch()

def lu_decomposicao(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        L[(k + 1):, k] = (A[(k + 1):, k] - L[(k + 1):, :] @ U[:, k]) / U[k, k]

    return np.eye(n), L, U
def Cholesky(stdscr): 
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz vazia
    matriz = np.zeros((ordem, ordem))
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz simétrica {ordem}x{ordem} (apenas a parte triangular inferior):\n")
    for i in range(ordem):
        for j in range(i + 1):  # Apenas a parte triangular inferior
            elemento = float(get_input(stdscr, f"Digite os elementos da matriz simétrica {ordem}x{ordem} (apenas a parte triangular inferior):\n" + f"Elemento [{i+1},{j+1}]: "))
            matriz[i, j] = elemento
            matriz[j, i] = elemento
    
    # Verifica se a matriz é positiva-definida
    try:
        np.linalg.cholesky(matriz)
    except np.linalg.LinAlgError:
        stdscr.addstr("A matriz não é positiva-definida. Encerrando...\n")
        curses.curs_set(0)
        stdscr.getch()
        return
    
    # Solicita o vetor dos termos independentes
    vetor = []
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr, f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" + f"Elemento [{i+1}]: "))
        vetor.append(elemento)
    
    # Converte o vetor para array numpy
    vetor_np = np.array(vetor)
    
    # Decomposição de Cholesky
    L = np.linalg.cholesky(matriz)
    
    # Resolve o sistema Ly = b
    y = np.linalg.solve(L, vetor_np)
    
    # Resolve o sistema L^T x = y
    x = np.linalg.solve(L.T, y)
    
    # Imprime a solução
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {x[i]:.4f}\n")
    
    curses.curs_set(0)
    stdscr.getch()
def GaussCompacto(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz estendida vazia
    matriz_estendida = []
    
    # Solicita os elementos da matriz estendida
    stdscr.addstr(f"Digite os elementos da matriz estendida {ordem}x{ordem+1}:\n")
    for i in range(ordem):
        linha = []
        for j in range(ordem + 1):
            elemento = float(get_input(stdscr, f"Digite os elementos da matriz estendida {ordem}x{ordem+1}:\n" + f"Elemento [{i+1},{j+1}]: "))
            linha.append(elemento)
        matriz_estendida.append(linha)
    
    # Converte a lista de listas em um array numpy
    matriz_estendida_np = np.array(matriz_estendida)
    

    # Aplica o método de Gauss compacto
    for k in range(ordem):  # Percorre as colunas
        for i in range(k + 1, ordem):  # Percorre as linhas abaixo da linha k
            if matriz_estendida_np[k, k] != 0:  # Evita divisão por zero
                fator = matriz_estendida_np[i, k] / matriz_estendida_np[k, k]
                matriz_estendida_np[i, k:] -= fator * matriz_estendida_np[k, k:]
    
    # Resolve o sistema triangular superior resultante
    solucao = np.zeros(ordem)
    for i in range(ordem - 1, -1, -1):  # Percorre as linhas de baixo para cima
        if matriz_estendida_np[i, i] != 0:  # Evita divisão por zero
            solucao[i] = (matriz_estendida_np[i, -1] - np.dot(matriz_estendida_np[i, i+1:-1], solucao[i+1:])) / matriz_estendida_np[i, i]
    
    # Imprime a solução
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {solucao[i]:.4f}\n")
    
    curses.curs_set(0)
    stdscr.getch()
def GaussJordan(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz dos coeficientes e o vetor dos termos independentes
    matriz = np.zeros((ordem, ordem))
    vetor = np.zeros(ordem)
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem}:\n")
    for i in range(ordem):
        for j in range(ordem):
            elemento = float(get_input(stdscr,f"Digite os elementos da matriz {ordem}x{ordem+1}:\n" +  f"Elemento [{i+1},{j+1}]: "))
            matriz[i, j] = elemento
    
    # Solicita os elementos do vetor dos termos independentes
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr,f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" + f"Elemento [{i+1}]: "))
        vetor[i] = elemento
    
    # Combina a matriz dos coeficientes com o vetor dos termos independentes
    matriz_expandida = np.column_stack((matriz, vetor))
    
    # Aplica o método de Gauss-Jordan
    for i in range(ordem):
        # Divide a linha pelo pivô (elemento diagonal principal)
        pivô = matriz_expandida[i, i]
        matriz_expandida[i] /= pivô
        
        # Zera os elementos abaixo e acima do pivô na coluna
        for j in range(ordem):
            if i != j:
                fator = matriz_expandida[j, i]
                matriz_expandida[j] -= fator * matriz_expandida[i]
    
    # Obtém o vetor solução a partir da última coluna da matriz expandida
    solucao = matriz_expandida[:, -1]
    
    # Imprime a solução
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {solucao[i]:.4f}\n")
    
    curses.curs_set(0)
    stdscr.getch()
def Jacobi(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz dos coeficientes, o vetor dos termos independentes,
    # a aproximação inicial para a solução, a precisão desejada e o número máximo de iterações
    matriz = np.zeros((ordem, ordem))
    vetor = np.zeros(ordem)
    aproximacao = np.zeros(ordem)
    precisao = float(get_input(stdscr, "Digite a precisão desejada (ε): "))
    max_iter = int(get_input(stdscr, "Digite o número máximo de iterações: "))
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem}:\n")
    for i in range(ordem):
        for j in range(ordem):
            elemento = float(get_input(stdscr, f"Digite os elementos da matriz {ordem}x{ordem}:\n" + f"Elemento [{i+1},{j+1}]: "))
            matriz[i, j] = elemento
    
    # Solicita os elementos do vetor dos termos independentes
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr, f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" + f"Elemento [{i+1}]: "))
        vetor[i] = elemento
    
    # Solicita a aproximação inicial para a solução
    stdscr.addstr("Digite a aproximação inicial para a solução:\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr, "Digite a aproximação inicial para a solução (Use . ao inves de ,):\n" + f"x[{i+1}]: "))
        aproximacao[i] = elemento
    
    # Aplica o método de Jacobi
    iteracao = 0
    while iteracao < max_iter:
        iteracao += 1
        nova_aproximacao = np.zeros(ordem)
        for i in range(ordem):
            soma = 0
            for j in range(ordem):
                if j != i:
                    soma += matriz[i, j] * aproximacao[j]
            nova_aproximacao[i] = (vetor[i] - soma) / matriz[i, i]
        
        # Calcula a norma da diferença entre a nova e a antiga aproximação
        norma_diferenca = np.linalg.norm(nova_aproximacao - aproximacao)
        
        # Atualiza a aproximação anterior com a nova aproximação
        aproximacao = nova_aproximacao.copy()
        
        # Verifica se a precisão foi atingida
        if norma_diferenca < precisao:
            break
    
    # Imprime a solução e o número de iteraçdef MetodoDecomposicaoLU(matriz):
    # Calcula a decomposição LU da matriz
    try:
        P, L, U = np.linalg.lu(matriz)
        return np.linalg.inv(U) @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        return Noneões
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {aproximacao[i]:.4f}\n")
    stdscr.addstr(f"Número de iterações: {iteracao}\n")
    
    curses.curs_set(0)
    stdscr.getch()
def GaussSeidel(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem do sistema: "))
    
    # Inicializa a matriz dos coeficientes, o vetor dos termos independentes,
    # a aproximação inicial para a solução, a precisão desejada e o número máximo de iterações
    matriz = np.zeros((ordem, ordem))
    vetor = np.zeros(ordem)
    aproximacao = np.zeros(ordem)
    precisao = float(get_input(stdscr, "Digite a precisão desejada (ε): "))
    max_iter = int(get_input(stdscr, "Digite o número máximo de iterações: "))
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem}:\n")
    for i in range(ordem):
        for j in range(ordem):
            elemento = float(get_input(stdscr,f"Digite os elementos da matriz {ordem}x{ordem}:\n" + f"Elemento [{i+1},{j+1}]: "))
            matriz[i, j] = elemento
    
    # Solicita os elementos do vetor dos termos independentes
    stdscr.addstr(f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr,f"Digite os elementos do vetor dos termos independentes (tamanho {ordem}):\n" + f"Elemento [{i+1}]: "))
        vetor[i] = elemento
    
    # Solicita a aproximação inicial para a solução
    stdscr.addstr("Digite a aproximação inicial para a solução:\n")
    for i in range(ordem):
        elemento = float(get_input(stdscr, "Digite a aproximação inicial para a solução:\n" + f"x[{i+1}]: "))
        aproximacao[i] = elemento
    
    # Aplica o método de Gauss-Seidel
    iteracao = 0
    while iteracao < max_iter:
        iteracao += 1
        nova_aproximacao = np.zeros(ordem)
        for i in range(ordem):
            soma1 = np.dot(matriz[i, :i], nova_aproximacao[:i])
            soma2 = np.dot(matriz[i, i + 1:], aproximacao[i + 1:])
            nova_aproximacao[i] = (vetor[i] - soma1 - soma2) / matriz[i, i]
        
        # Calcula a norma da diferença entre a nova e a antiga aproximação
        norma_diferenca = np.linalg.norm(nova_aproximacao - aproximacao)
        
        # Atualiza a aproximação anterior com a nova aproximação
        aproximacao = nova_aproximacao.copy()
        
        # Verifica se a precisão foi atingida
        if norma_diferenca < precisao:
            break
    
    # Imprime a solução e o número de iterações
    stdscr.addstr("Vetor solução:\n")
    for i in range(ordem):
        stdscr.addstr(f"x[{i+1}] = {aproximacao[i]:.4f}\n")
    stdscr.addstr(f"Número de iterações: {iteracao}\n")
    
    curses.curs_set(0)
    stdscr.getch()

def MetodoDecomposicaoLU(matriz):
    n = len(matriz)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1  # Diagonal principal de L é 1
        for j in range(i, n):
            soma = sum(U[k, j] * L[i, k] for k in range(i))
            U[i, j] = matriz[i, j] - soma

        for j in range(i + 1, n):
            soma = sum(U[k, i] * L[j, k] for k in range(i))
            L[j, i] = (matriz[j, i] - soma) / U[i, i]

def MetodoGaussCompacto(matriz):
    # Calcula a matriz inversa usando o método de Gauss Compacto
    ordem = matriz.shape[0]
    matriz_expandida = np.column_stack((matriz, np.identity(ordem)))
    
    for i in range(ordem):
        # Normaliza a linha i (elemento diagonal principal igual a 1)
        divisor = matriz_expandida[i, i]
        matriz_expandida[i] /= divisor
        
        # Zera os elementos abaixo e acima do pivô na coluna
        for j in range(ordem):
            if i != j:
                fator = matriz_expandida[j, i]
                matriz_expandida[j] -= fator * matriz_expandida[i]
    
    return matriz_expandida[:, ordem:]

def MatrizInversa(stdscr):
    # Solicita a ordem da matriz
    ordem = int(get_input(stdscr, "Digite a ordem da matriz: "))
    
    # Inicializa a matriz vazia
    matriz = np.zeros((ordem, ordem))
    
    # Solicita os elementos da matriz
    stdscr.addstr(f"Digite os elementos da matriz {ordem}x{ordem}:\n")
    for i in range(ordem):
        for j in range(ordem):
            elemento = float(get_input(stdscr, f"Elemento [{i+1},{j+1}]: "))
            matriz[i, j] = elemento
    
    # Solicita o método a ser utilizado
    metodo = get_input(stdscr, "Digite o método a ser utilizado (\"lu\" ou \"gausscompacto\"): ").lower()
    
    # Calcula a matriz inversa de acordo com o método escolhido
    if metodo == 'lu':
        inversa = MetodoDecomposicaoLU(matriz)
    elif metodo == 'gausscompacto':
        inversa = MetodoGaussCompacto(matriz)
    else:
        stdscr.addstr("Método inválido. Encerrando...\n")
        curses.curs_set(0)
        stdscr.getch()
        return
    
    # Verifica se a matriz inversa foi calculada com sucesso
    if inversa is not None:
        # Imprime a matriz inversa
        stdscr.addstr("Matriz inversa:\n")
        for linha in inversa:
            for elemento in linha:
                stdscr.addstr(f"{elemento:.4f} ")
            stdscr.addstr("\n")
    else:
        stdscr.addstr("Não foi possível calcular a matriz inversa. Encerrando...\n")
    
    curses.curs_set(0)
    stdscr.getch()

curses.wrapper(main)