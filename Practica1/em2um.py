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

    - q: numero real, carga
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

    - q: numero real, carga
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