"""
em2um.py - Biblioteca de Electromagnetismo 2
                Universidad de Murcia

Created on Tue Feb 25 09:58:51 2025
@author: gregomc
"""
import numpy as np

# Constantes universales
pi = np.pi
E_0 = 8.8541878188e-12  # permitividad vacio
k_0 = 1/(4 * pi * E_0) # constante de Coulomb
MU_0 = 4 * pi * 1e-7  # permeabilidad vacio


def potencial_carga(q, pos, x, y):
    """
    Calcula el potencial electrico de una carga puntual.
    Desarrollada en la practica 1

    - q: float, carga
    - pos: array de dos elementos, posición de la carga
    - x,y: array/matrices, puntos donde evaluar el potencial
    """
    dx = x - pos[0]
    dy = y - pos[1]
    r = np.sqrt(dx ** 2 + dy ** 2)
    r = np.where(r < 1e-10, 1e-10, r) # Evitar division por cero
    return q * k_0 / r


def campo_carga(q, pos, x, y):
    """
    Calcula el campo electrico de una carga puntual.
    Desarrollada en P1

    - q: float, carga
    - pos: array de dos elementos, posición de la carga
    - x,y: array/matrices, puntos donde evaluar el campo
    """
    dx = x - pos[0]
    dy = y - pos[1]
    r_cubo = (dx ** 2 + dy ** 2) ** (3 / 2)
    r_cubo = np.where(r_cubo < 1e-10, 1e-10, r_cubo) # Evitar division por cero

    Ex = k_0 * q * dx / r_cubo
    Ey = k_0 * q * dy / r_cubo
    return Ex, Ey


import numpy as np


# Incluir en la libreria em2um
def SemiCajaPotencial(NumModos, V0, L, x, y):
    """
    Calcula el potencial eléctrico (Pot) para una semicaja.
    Desarrollada en P2

    - NumModos: int, numero de sumandos de la serie
    - V0: float, condicion de contorno
    - L: float, longitud del ancho de la semicaja donde esta V0
    - x: array/matrices, puntos donde evaluar el potencial
    - y: array/matrices, puntos donde evaluar el campo
    """
    if NumModos < 1:
        return np.NaN

    # Genera una lista de modos impares: 1, 3, 5, ..., 2*NumModos - 1
    Modos = np.arange(1, 2 * NumModos, 2)
    Pot = 0
    for m in Modos:
        Pot += V0 * 4 / (pi * m) * np.sin(x * m * pi / L) * np.exp(-y * m * pi / L)
    return Pot


def SemiCajaCampo(NumModos, V0, L, x, y):
    """
    Calcula los componentes Ex y Ey del campo eléctrico para una semicaja.
    Desarrollada en P2

    - NumModos: int, numero de sumandos de la serie
    - V0: float, condicion de contorno
    - L: float, longitud del ancho de la semicaja donde esta V0
    - x: array/matrices, puntos donde evaluar el potencial
    - y: array/matrices, puntos donde evaluar el campo
    """
    if NumModos < 1:
        return np.NaN, np.NaN

    # Genera una lista de modos impares: 1, 3, 5, ..., 2*NumModos - 1
    Modos = np.arange(1, 2 * NumModos, 2)
    Ex = 0
    Ey = 0

    for m in Modos:
        arg_x = x * m * pi / L
        arg_y = - y * m * pi / L
        Ex -= V0 * 4 / L * np.cos(arg_x) * np.exp(arg_y)
        Ey += V0 * 4 / L * np.sin(arg_x) * np.exp(arg_y)
    return Ex, Ey

