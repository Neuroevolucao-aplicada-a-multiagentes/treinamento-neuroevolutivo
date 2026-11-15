"""Classifica, cenario a cenario, por que um controlador nao entregou.

Usa os cenarios fixos de avaliar_checkpoints.py e olha os ultimos segundos de
cada cenario sem entrega:

    eliminado           atingiu o limite de colisoes
    preso atras         quase parado, com uma parede entre ele e o alvo
    preso com visada    quase parado, sem parede entre ele e o alvo
    vagando atras       ainda se movendo, com uma parede entre ele e o alvo
    vagando com visada  ainda se movendo, sem parede entre ele e o alvo

Uso:
    python src/diagnosticar_falhas.py melhor_rede_fase5.npz campos
    python src/diagnosticar_falhas.py --ambientes fase5 corredores melhor_rede_fase5.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from avaliar_checkpoints import AMBIENTES, _RAIZ, _config, _controlador, _expandir, _rotulo

JANELA = 10.0       # segundos finais observados
QUASE_PARADO = 40.0  # px percorridos na janela abaixo dos quais o agente esta preso

CLASSES = ("eliminado", "preso atras", "preso com visada", "vagando atras", "vagando com visada")


def _bloqueado(ambiente, origem, destino) -> bool:
    return any(obs.clipline((origem.x, origem.y), (destino.x, destino.y))
               for obs in ambiente.obstaculos)


def _classificar(tarefa: tuple[str, str, int, int]) -> dict:
    caminho, nome_ambiente, n, semente_base = tarefa
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from simulador import Agente, Ambiente, rodar_geracao

    cfg = _config(nome_ambiente)
    rede = _controlador(caminho)
    dt = 1.0 / 60.0
    passos_janela = int(JANELA / dt)
    linhas = []
    for k in range(n):
        amb = Ambiente(cfg, rng=random.Random(semente_base + k))
        agente = Agente(rede.copy(), amb)
        agente.reset_estado()
        agente.pos = pygame.Vector2(amb.start_pos)
        trilha = []

        def registrar(passo, tempo):
            trilha.append(pygame.Vector2(agente.pos))
            return False

        rodar_geracao([agente], amb, cfg, dt_fixo=dt, on_step=registrar)
        if agente.entregou:
            classe = "entregou"
        elif agente.morto:
            classe = "eliminado"
        else:
            recente = trilha[-passos_janela:]
            percorrido = sum(a.distance_to(b) for a, b in zip(recente, recente[1:]))
            estado = "preso" if percorrido < QUASE_PARADO else "vagando"
            lado = "atras" if _bloqueado(amb, agente.pos, agente.alvo_atual()) else "com visada"
            classe = f"{estado} {lado}"
        linhas.append({"cenario": k, "semente": semente_base + k, "classe": classe,
                       "coletou": bool(agente.coletou)})
    return {"caminho": caminho, "ambiente": nome_ambiente, "linhas": linhas}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("redes", nargs="+")
    parser.add_argument("--cenarios", type=int, default=100)
    parser.add_argument("--ambientes", nargs="+", choices=AMBIENTES, default=["corredores"])
    parser.add_argument("--semente-base", type=int, default=9000)
    parser.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    redes = _expandir(args.redes)
    tarefas = [(r, a, args.cenarios, args.semente_base) for a in args.ambientes for r in redes]
    with ProcessPoolExecutor(max_workers=args.processos) as pool:
        resultados = list(pool.map(_classificar, tarefas))

    linhas_csv = []
    for res in resultados:
        contagem = Counter(l["classe"] for l in res["linhas"])
        falhas = sum(v for c, v in contagem.items() if c != "entregou")
        print(f"\n=== {_rotulo(res['caminho'])} em {res['ambiente']}: "
              f"{falhas} falhas em {len(res['linhas'])} cenarios ===")
        for classe in CLASSES:
            antes = sum(1 for l in res["linhas"] if l["classe"] == classe and not l["coletou"])
            print(f"  {classe:<20} {contagem[classe]:3d}   (antes da coleta: {antes})")
        for l in res["linhas"]:
            linhas_csv.append({"rede": _rotulo(res["caminho"]), "ambiente": res["ambiente"], **l})

    pasta = os.path.join(_RAIZ, "runs", "avaliacoes")
    os.makedirs(pasta, exist_ok=True)
    saida = os.path.join(pasta, f"falhas_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas_csv)
    print(f"\n[falhas] tabela gravada em {os.path.relpath(saida, _RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
