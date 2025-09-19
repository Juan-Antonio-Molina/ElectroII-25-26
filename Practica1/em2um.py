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
MU_0 = 4 * pi * 1e-7  # permeabilidad vacio


###
def potencial_carga(q, pos, x, y):
    """
    Calcula el potencial electrico de una carga puntual.
    Desarrollada en la practica 1

    - pos: array de dos elementos

    Ejemplo:
    # >>> potencial_carga(1e-9, [0, 0], 1, 0)
    8.987551787368176e-01 # = 1e-9/(4*PI*E_0*1*1), 0)
    """
    dx = x - pos[0]
    dy = y - pos[1]
    r = np.sqrt(dx ** 2 + dy ** 2)

    # Evitar division por cero
    r = np.where(r < 1e-10, 1e-10, r)
    return q / (4 * pi * E_0) / r


def campo_carga(q, pos, x, y):
    """
    Calcula el campo electrico de una carga puntual.
    Desarrollada en P1

    - pos: array de dos elementos

    Ejemplo:
    # >>> campo_carga(1e-9, [0, 0], 1, 0)
    (8.987551787368176e-01, 0.0) # = (1e-9/(4*PI*E_0*1*1), 0)
    """
    dx = x - pos[0]
    dy = y - pos[1]
    r_cubo = (dx ** 2 + dy ** 2) ** (3 / 2)

    # Evitar division por cero
    r_cubo = np.where(r_cubo < 1e-10, 1e-10, r_cubo)

    Ex = 1 / (4 * pi * E_0) * q * dx / r_cubo
    Ey = 1 / (4 * pi * E_0) * q * dy / r_cubo
    return Ex, Ey