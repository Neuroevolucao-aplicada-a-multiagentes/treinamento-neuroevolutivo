"""Controlador de referencia por campos potenciais (KHATIB, 1986).

Recebe o mesmo vetor de 16 entradas da rede e devolve o mesmo vetor de
velocidade. O alvo atrai com forca unitaria, e cada raio que encontra um
obstaculo dentro da fracao `alcance` do sensor repele com forca que cresce com
o quadrado da proximidade. Como os raios saem da direcao de movimento, o
controlador guarda o ultimo comando.
"""

from __future__ import annotations

import math

import numpy as np

NUM_RAYS = 8


class CamposPotenciais:
    def __init__(self, ganho: float = 1.5, alcance: float = 0.5):
        self.ganho = ganho
        self.alcance = alcance
        self._ultimo = np.zeros(2, dtype=np.float32)

    def copy(self) -> "CamposPotenciais":
        return CamposPotenciais(self.ganho, self.alcance)

    def forward(self, x) -> np.ndarray:
        alvo = np.array([x[0], x[1]], dtype=np.float64)
        if np.linalg.norm(self._ultimo) > 0.01:
            heading = math.atan2(self._ultimo[1], self._ultimo[0])
        else:
            heading = math.atan2(alvo[1], alvo[0])
        forca = alvo.copy()
        for i in range(NUM_RAYS):
            leitura = float(x[8 + i])
            if leitura >= self.alcance:
                continue
            ang = heading + (i / NUM_RAYS) * 2 * math.pi
            proximidade = 1.0 - leitura / self.alcance
            forca -= self.ganho * proximidade ** 2 * np.array([math.cos(ang), math.sin(ang)])
        norma = np.linalg.norm(forca)
        saida = (forca / norma) if norma > 1e-6 else alvo
        self._ultimo = saida.astype(np.float32)
        return self._ultimo
