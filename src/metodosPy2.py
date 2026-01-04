# TRABALHO 2 DE MÉTODOS NUMERICOS
# FEITO POR: KAUÃ JUNIOR SILVA SOARES - 231024061



import numpy as np

def menu():  #MENU
    while True:
        print("\nMenu:")
        print("1. Função: Retorna o valor do polinômio interpolado (Newton)")
        print("2. Função: Retorna o valor do polinômio interpolado (Newton-Gregory)")
        print("3. Função: Retorna o coeficiente de determinação")
        print("4. Função: Ajusta os pontos tabelados a uma reta")
        print("5. Função: Ajusta os pontos tabelados a um polinômio de grau desejado")
        print("6. Função: Ajusta os pontos tabelados a uma curva exponencial")
        print("0. Sair")

        choice = int(input("Escolha uma opção: "))

        if choice == 1:
            n, tabela, x = parametrosNewton()
            resultado = newton(n, tabela, x)
            print(f"Valor do polinômio interpolado: {resultado}")
        elif choice == 2:

            n, tabela, x = parametrosNewton()
            resultado = newtonGregory(n, tabela, x)
            print(f"Valor do polinômio interpolado: {resultado}")
        elif choice == 3:
            n, tabela = parametrosPontos()
            resultado, coeficiente = coeficienteDeterminacao(n, tabela)
            print(f"Valores ajustados: {resultado}")
            print(f"Coeficiente de determinação: {coeficiente}")
        elif choice == 4:
            n, tabela = parametrosPontos()
            a0, a1, Yajustado, coeficiente = AjusteReta(n, tabela)
            print(f"Termo independente (a0): {a0}")
            print(f"Coeficiente de grau 1 (a1): {a1}")
            print(f"Valores ajustados: {Yajustado}")
            print(f"Coeficiente de determinação: {coeficiente}")
        elif choice == 5:
            n, grau, tabela = parametrosPolinômio()
            coeficientes, Yajustado, coeficiente = AjustePolinomio(n, grau, tabela)
            print(f"Coeficientes do polinômio ajustado: {coeficientes}")
            print(f"Valores ajustados: {Yajustado}")
            print(f"Coeficiente de determinação: {coeficiente}")
        elif choice == 6:
            n, tabela = parametrosPontos()
            a, b, Yajustado, coeficiente = AjusteExponencial(n, tabela)
            print(f"1º coeficiente (a): {a}")
            print(f"2º coeficiente (b): {b}")
            print(f"Valores ajustados: {Yajustado}")
            print(f"Coeficiente de determinação: {coeficiente}")
        elif choice == 0:
            break
        else:
            print("Opção inválida, tente novamente.")


# LEITURA DOS PARÂMETROS

def parametrosNewton():
    n = int(input("Número de pontos tabelados: "))
    tabela = []
    for q in range(n):
        x, y = map(float, input("Digite x e y separados por espaço (um ponto de cada vez): ").split())
        tabela.append((x, y))
    x = float(input("Ponto onde se deseja conhecer o P(x): "))
    return n, tabela, x

def parametrosPontos():
    n = int(input("Número de pontos tabelados: "))
    tabela = []
    for q in range(n):
        x, y = map(float, input("Digite x e y separados por espaço (um ponto de cada vez): ").split())
        tabela.append((x, y))
    return n, tabela

def parametrosPolinômio():
    n = int(input("Número de pontos tabelados: "))
    grau = int(input("Grau desejado do polinômio: "))
    tabela = []
    for q in range(n):
        x, y = map(float, input("Digite x e y separados por espaço (um ponto de cada vez): ").split())
        tabela.append((x, y))
    return n, grau, tabela

# FUNÇÕES

def newton(n, tabela, x):
    # Função para calcular as diferenças divididas
    def coefDasDiferencasDivididas(tabela):
        n = len(tabela)
        coef = [tabela[i][1] for i in range(n)]
        for j in range(1, n):
            for i in range(n-1, j-1, -1):
                coef[i] = (coef[i] - coef[i-1]) / (tabela[i][0] - tabela[i-j][0])
        return coef
    # Calcula os coeficientes das diferenças divididas
    coef = coefDasDiferencasDivididas(tabela)
    n = len(tabela)
    resultado = coef[-1]
    # Avalia o polinômio interpolado usando os coeficientes
    for i in range(n-2, -1, -1):
        resultado = resultado * (x - tabela[i][0]) + coef[i]
    return resultado

def newtonGregory(n, tabela, x):
    # Função para calcular as diferenças progressivas
    def diferencasFinitas(tabela):
        n = len(tabela)
        tabelax = [[0]*n for varN in range(n)]
        for i in range(n):
            tabelax[i][0] = tabela[i][1]
        for j in range(1, n):
            for i in range(n-j):
                tabelax[i][j] = tabelax[i+1][j-1] - tabelax[i][j-1]
        return [tabelax[0][i] for i in range(n)]
    # Função para calcular o produto dos termos (x - xi)
    def termoDoProduto(i, x, tabela):
        prod = 1
        for j in range(i):
            prod *= (x - tabela[j][0])
        return prod

    h = tabela[1][0] - tabela[0][0] # Calcula o espaçamento dos pontos
    b = diferencasFinitas(tabela) # Calcula as diferenças progressivas
    n = len(tabela)
    resultado = tabela[0][1]
    for i in range(1, n):     # Avalia o polinômio interpolado usando as diferenças progressivas
        resultado += (difadof[i] * termoDoProduto(i, x, tabela)) / (h**i)
    return resultado

def coeficienteDeterminacao(n, tabela):
    Ytabelado = [tabela[i][1] for i in range(n)]
    Ymedio = sum(Ytabelado) / n
    # Função para calcular a soma dos quadrados totais
    def somaTotal(Ytabelado, Ymedio):
        return sum((yi - Ymedio) ** 2 for yi in Ytabelado)
    # Função para calcular a soma dos quadrados residuais
    def SomaResidual(Ytabelado, Yprevisto):
        return sum((yi - ypi) ** 2 for yi, ypi in zip(Ytabelado, Yprevisto))
    # assume um ajuste linear para este exemplo
    a0, a1, Yprevisto = AjusteReta(n, tabela)[:3]

    sTotal = somaTotal(Ytabelado, Ymedio) # Calcula a soma dos quadrados totais
    sResi = SomaResidual(Ytabelado, Yprevisto) # Calcula a soma dos quadrados residuais

    coefDeDet = 1 - (sResi / sTotal) # Calcula o coeficiente de determinação
    return Yprevisto, coefDeDet

def AjusteReta(n, tabela):
    SomaX = SomaY = SomaXY = SomaX2 = 0
    for x, y in tabela:
        SomaX += x
        SomaY += y
        SomaXY += x * y
        SomaX2 += x * x
 # Calcula os coeficientes da reta ajustada
    a1 = (n * SomaXY - SomaX * SomaY) / (n * SomaX2 - SomaX * SomaX)
    a0 = (SomaY - a1 * SomaX) / n

    Yajustados = [a0 + a1 * x for x, varN in tabela]
# Calcula os valores ajustados
    varN, cofDeDet = coeficienteDeterminacao(n, tabela)

    return a0, a1, Yajustados, cofDeDet


def AjustePolinomio(n, grau, tabela):
    # Função para resolver um sistema linear usando eliminação de Gauss-Jordan
    def gaussJordan(m, aug):
        n = len(aug)
        for i in range(n):
            maxElem = abs(aug[i][i])
            maxLinh = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > maxElem:
                    maxElem = abs(aug[k][i])
                    maxLinh = k

            for k in range(i, n + 1):
                aug[maxLinh][k], aug[i][k] = aug[i][k], aug[maxLinh][k]

            for k in range(i + 1, n):
                c = -aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    if i == j:
                        aug[k][j] = 0
                    else:
                        aug[k][j] += c * aug[i][j]

        coef = [0 for i in range(n)]
        for i in range(n - 1, -1, -1):
            coef[i] = aug[i][n] / aug[i][i]
            for k in range(i - 1, -1, -1):
                aug[k][n] -= aug[k][i] * coef[i]
        return coef
    # Monta as matrizes X e Y para o ajuste polinomial
    X = [[x ** i for i in range(grau + 1)] for x, varN in tabela]
    Y = [y for varN, y in tabela]
    # Calcula a transposta de X
    XT = [[X[j][i] for j in range(n)] for i in range(grau + 1)]
    # Calcula o produto XT * X e XT * Y
    XTX = [[sum(XT[i][k] * X[k][j] for k in range(n)) for j in range(grau + 1)] for i in range(grau + 1)]
    XTY = [sum(XT[i][j] * Y[j] for j in range(n)) for i in range(grau + 1)]
    # Monta a matriz aumentada para o sistema linear
    aug = [XTX[i] + [XTY[i]] for i in range(grau + 1)]
    # Resolve o sistema linear usando Gauss-Jordan
    coef = gaussJordan(grau + 1, aug)
    # Calcula os valores ajustados
    Yajustados = [sum(coef[j] * (x ** j) for j in range(grau + 1)) for x, varN in tabela]

    varN, cofDeDet = coeficienteDeterminacao(n, tabela) # Calcula o coeficiente de determinação

    return coef, Yajustados, cofDeDet

def AjusteExponencial(n, tabela):
    # Aplica logaritmo natural aos valores de y
    lnTabela = [(x, np.log(y)) for x, y in tabela]
    # Ajusta os pontos transformados a uma reta
    a0, a1, lnYajustados, varN = AjusteReta(n, lnTabela)
    # Calcula os coeficientes da curva exponencial
    a = np.exp(a0)
    b = np.exp(a1)
    # Calcula os valores ajustados
    Yajustados = [a * (b ** x) for x, varN in tabela]

    varN, cofDeDet = coeficienteDeterminacao(n, tabela) # Calcula o coeficiente de determinação

    return a, b, Yajustados, cofDeDet

menu()