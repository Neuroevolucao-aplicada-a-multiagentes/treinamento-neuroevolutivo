"""Mede, em operacao continua, se uma rede sai sozinha de um contato.

Varios robos operam sem parar e sem eliminacao, e cada um recebe uma tarefa
nova ao entregar. Cada rede roda com e sem a camada de recuperacao do armazem
3D (scripts/ia/agente_ia.gd).

Uso:
    python src/medir_encalhe.py melhor_rede_fase5.npz "runs/*G_contato*/melhor_rede.npz"
    python src/medir_encalhe.py --robos 8 --duracao 140 --sorteios 10 melhor_rede_fase5.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from avaliar_checkpoints import (AMBIENTES, FRACAO_AVANCO, TEMPO_ATE_ENCALHE, _RAIZ, _config,
                                 _controlador, _expandir, _rotulo)

DURACAO_DESENCALHE = 0.6


class _ComDesencalhe:
    """Assume o comando do robo enquanto a camada de recuperacao age."""

    def __init__(self, brain):
        self.brain = brain
        self.tempo = 0.0
        self.direcao = None

    def forward(self, x):
        import numpy as np
        if self.tempo > 0 and self.direcao is not None:
            return np.array([self.direcao.x, self.direcao.y], dtype=np.float32)
        return self.brain.forward(x)


def _sinal(v: float) -> float:
    return (v > 0) - (v < 0)


def _operar(tarefa: tuple[str, str, int, float, int, bool]) -> dict:
    caminho, nome_ambiente, num_robos, duracao, semente, recuperacao = tarefa
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from simulador import VELOCIDADE, Ambiente, rodar_operacao_multiagente

    cfg = _config(nome_ambiente)
    rede = _controlador(caminho)
    ambiente = Ambiente(cfg, rng=random.Random(semente))
    dt = 1.0 / 60.0
    avanco_minimo = FRACAO_AVANCO * VELOCIDADE * dt
    anteriores: dict[int, pygame.Vector2] = {}
    sem_avancar: dict[int, float] = {}
    contagem = {"encalhado": 0.0, "desencalhes": 0}

    def acompanhar(passo, tempo, robos):
        for r in robos:
            if passo == 0 and recuperacao:
                r.brain = _ComDesencalhe(r.brain)
            anterior = anteriores.get(r.id, pygame.Vector2(r.pos))
            avancou = r.pos.distance_to(anterior) >= avanco_minimo
            em_contato = r.em_colisao_obs or r.em_colisao_robo
            if em_contato and not avancou:
                contagem["encalhado"] += dt

            camada = r.brain if recuperacao else None
            if camada is not None and camada.tempo > 0:
                camada.tempo -= dt
                sem_avancar[r.id] = 0.0
            elif avancou:
                sem_avancar[r.id] = 0.0
            else:
                sem_avancar[r.id] = sem_avancar.get(r.id, 0.0) + dt
                if camada is not None and sem_avancar[r.id] >= TEMPO_ATE_ENCALHE:
                    sem_avancar[r.id] = 0.0
                    # Normal da parede: o eixo em que o comando nao virou movimento.
                    bloqueado_x = abs(r.vel.x) > 1e-3 and abs(r.pos.x - anterior.x) < 1e-6
                    bloqueado_y = abs(r.vel.y) > 1e-3 and abs(r.pos.y - anterior.y) < 1e-6
                    normal = pygame.Vector2(-_sinal(r.vel.x) if bloqueado_x else 0.0,
                                            -_sinal(r.vel.y) if bloqueado_y else 0.0)
                    if normal.length() > 0:
                        normal = normal.normalize()
                        tangente = pygame.Vector2(-normal.y, normal.x)
                        if tangente.dot(r.alvo_atual() - r.pos) < 0:
                            tangente = -tangente
                        camada.direcao = (tangente * 0.8 + normal * 0.6).normalize()
                        camada.tempo = DURACAO_DESENCALHE
                        contagem["desencalhes"] += 1
            anteriores[r.id] = pygame.Vector2(r.pos)
        return False

    robos, metricas = rodar_operacao_multiagente(rede, num_robos, ambiente,
                                                  duracao=duracao, dt=dt, on_step=acompanhar)
    return {
        "caminho": caminho, "ambiente": nome_ambiente, "semente": semente,
        "recuperacao": recuperacao,
        "entregas": metricas["total_entregas"],
        "robos_sem_entrega": sum(1 for r in robos if r.total_entregas == 0),
        "fracao_encalhado": contagem["encalhado"] / (duracao * num_robos),
        "desencalhes": contagem["desencalhes"],
        "colisoes_obstaculo": metricas["colisoes_obstaculo"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("redes", nargs="+")
    parser.add_argument("--ambientes", nargs="+", choices=AMBIENTES, default=list(AMBIENTES))
    parser.add_argument("--robos", type=int, default=8)
    parser.add_argument("--duracao", type=float, default=140.0)
    parser.add_argument("--sorteios", type=int, default=10)
    parser.add_argument("--semente-base", type=int, default=9500)
    parser.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    redes = _expandir(args.redes)
    tarefas = [(r, a, args.robos, args.duracao, args.semente_base + k, rec)
               for a in args.ambientes for r in redes
               for rec in (True, False) for k in range(args.sorteios)]
    print(f"[encalhe] {len(redes)} redes x {len(args.ambientes)} ambientes x {args.sorteios} sorteios, "
          f"{args.robos} robos por {args.duracao:.0f} s, com e sem recuperacao")
    with ProcessPoolExecutor(max_workers=args.processos) as pool:
        resultados = list(pool.map(_operar, tarefas))

    for ambiente in args.ambientes:
        print(f"\n=== {ambiente} ===")
        print(f"{'rede':<28} {'recuperacao':>11} {'entregas':>14} {'sem entrega':>12} "
              f"{'encalhado':>10} {'desencalhes':>12}")
        for rede in redes:
            for rec in (True, False):
                linhas = [r for r in resultados
                          if r["caminho"] == rede and r["ambiente"] == ambiente and r["recuperacao"] == rec]
                entregas = [r["entregas"] for r in linhas]
                print(f"{_rotulo(rede):<28} {'com' if rec else 'sem':>11} "
                      f"{statistics.mean(entregas):7.1f} +-{statistics.pstdev(entregas):5.1f} "
                      f"{statistics.mean(r['robos_sem_entrega'] for r in linhas):12.1f} "
                      f"{statistics.mean(r['fracao_encalhado'] for r in linhas) * 100:9.1f}% "
                      f"{statistics.mean(r['desencalhes'] for r in linhas):12.1f}")

    pasta = os.path.join(_RAIZ, "runs", "avaliacoes")
    os.makedirs(pasta, exist_ok=True)
    saida = os.path.join(pasta, f"encalhe_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["rede", *resultados[0].keys()])
        escritor.writeheader()
        for r in resultados:
            escritor.writerow({"rede": _rotulo(r["caminho"]), **r})
    print(f"\n[encalhe] tabela gravada em {os.path.relpath(saida, _RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
