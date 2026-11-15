"""Avalia redes treinadas no mesmo conjunto fixo de cenarios.

Todas as redes enfrentam os mesmos cenarios, gerados por sementes fixas e
distintas das do treino, e a comparacao entre duas redes e pareada.

Ambientes:
    fase5       armazem da fase 5, com obstaculos e robos moveis
    corredores  topologia de corredores, fora do treino

Uso:
    python src/avaliar_checkpoints.py melhor_rede_fase5.npz "runs/Fase_5_*2026-10-13*/melhor_rede.npz"
    python src/avaliar_checkpoints.py --cenarios 60 --ambientes corredores melhor_rede_fase5.npz
    python src/avaliar_checkpoints.py melhor_rede_fase5.npz campos campos:2.0:0.4

"campos" avalia o controlador por campos potenciais; "campos:G:A" fixa o ganho
de repulsao G e o alcance A. --ruido S soma ruido gaussiano de desvio S aos
raios e --sem-eliminacao ignora o limite de colisoes. A primeira rede da lista
e a referencia das comparacoes pareadas.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AMBIENTES = ("fase5", "corredores")

# Encalhe: em contato e andando menos de 20% do passo maximo por 0,5 s,
# o mesmo criterio da camada de recuperacao do armazem 3D.
FRACAO_AVANCO = 0.2
TEMPO_ATE_ENCALHE = 0.5


def _config(ambiente: str):
    if ambiente == "fase5":
        import config_fase5
        cfg = config_fase5.CONFIG
    elif ambiente == "corredores":
        import config_fase6
        cfg = config_fase6.CONFIG
        cfg.layouts_fixos = 0
    else:
        raise ValueError(ambiente)
    cfg.cenarios_por_geracao = 1
    return cfg


def _rotulo(caminho: str) -> str:
    if _eh_campos(caminho):
        partes = caminho.split(":")
        return "campos potenciais" if len(partes) == 1 else f"campos g={partes[1]} a={partes[2]}"
    pasta = os.path.dirname(os.path.abspath(caminho))
    config_txt = os.path.join(pasta, "config.txt")
    if os.path.basename(pasta).startswith("Fase_") and os.path.exists(config_txt):
        # Runs com rotulo tem um segundo parenteses no nome da pasta.
        grupos = re.findall(r"\(([^)]*)\)", os.path.basename(pasta))
        variante = grupos[-1].replace("_", " ") + " " if len(grupos) > 1 else ""
        with open(config_txt, encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("seed ="):
                    return f"{variante}semente {linha.split('=', 1)[1].strip()}"
        return os.path.basename(pasta)
    return os.path.splitext(os.path.basename(caminho))[0]


def _eh_campos(caminho: str) -> bool:
    return caminho.split(":")[0] == "campos"


def _controlador(caminho: str):
    if _eh_campos(caminho):
        from baseline_campos_potenciais import CamposPotenciais
        return CamposPotenciais(*(float(p) for p in caminho.split(":")[1:]))
    from rede_transfer import carregar_rede_base
    return carregar_rede_base(caminho, input_size=16)


class _ComRuido:
    """Entrega ao controlador as leituras dos raios com ruido gaussiano."""

    def __init__(self, controlador, sigma: float, semente: int):
        import numpy as np
        self.controlador = controlador
        self.sigma = sigma
        self.rng = np.random.default_rng(semente)

    def copy(self) -> "_ComRuido":
        return self

    def forward(self, x):
        import numpy as np
        x = np.array(x, dtype=np.float32)
        x[8:16] = np.clip(x[8:16] + self.rng.normal(0.0, self.sigma, 8), 0.0, 1.0)
        return self.controlador.forward(x)


def _avaliar(tarefa: tuple[str, str, int, int, float, bool]) -> dict:
    """Roda uma rede em n cenarios de um ambiente. Executa num processo proprio."""
    caminho, ambiente, n, semente_base, ruido, sem_eliminacao = tarefa
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from simulador import VELOCIDADE, Agente, Ambiente, rodar_geracao

    cfg = _config(ambiente)
    if sem_eliminacao:
        cfg.max_colisoes_morte = 10**9
    rede = _controlador(caminho)
    dt = 1.0 / 60.0
    avanco_minimo = FRACAO_AVANCO * VELOCIDADE * dt
    por_cenario = []
    for k in range(n):
        amb = Ambiente(cfg, rng=random.Random(semente_base + k))
        controlador = rede.copy()
        if ruido > 0:
            controlador = _ComRuido(controlador, ruido, semente_base + k)
        agente = Agente(controlador, amb)
        agente.reset_estado()
        agente.pos = pygame.Vector2(amb.start_pos)
        estado = {"anterior": pygame.Vector2(agente.pos), "seguido": 0.0,
                  "encalhado": 0.0, "encalhes": 0}

        def medir_encalhe(passo, tempo):
            if agente.entregou or agente.morto:
                return False
            parado = agente.pos.distance_to(estado["anterior"]) < avanco_minimo
            estado["anterior"] = pygame.Vector2(agente.pos)
            if agente.em_colisao and parado:
                estado["seguido"] += dt
                if estado["seguido"] >= TEMPO_ATE_ENCALHE:
                    estado["encalhado"] += dt
                    if estado["seguido"] - dt < TEMPO_ATE_ENCALHE:
                        estado["encalhes"] += 1
            else:
                estado["seguido"] = 0.0
            return False

        rodar_geracao([agente], amb, cfg, dt_fixo=dt, on_step=medir_encalhe)
        por_cenario.append({
            "coletou": bool(agente.coletou),
            "entregou": bool(agente.entregou),
            "colisoes": int(agente.colisoes),
            "tempo_entrega": agente.tempo_entrega,
            "encalhes": estado["encalhes"],
            "tempo_encalhado": estado["encalhado"],
            "termina_encalhado": not agente.entregou and estado["seguido"] >= TEMPO_ATE_ENCALHE,
        })
    return {"caminho": caminho, "ambiente": ambiente, "cenarios": por_cenario}


def _wilson(sucessos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = sucessos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margem = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - margem), min(1.0, centro + margem)


def _mcnemar_exato(so_ref: int, so_cand: int) -> float:
    """p bilateral do teste exato de McNemar para resultados pareados binarios."""
    discordantes = so_ref + so_cand
    if discordantes == 0:
        return 1.0
    menor = min(so_ref, so_cand)
    cauda = sum(math.comb(discordantes, i) for i in range(menor + 1)) / 2 ** discordantes
    return min(1.0, 2 * cauda)


def _resumo(cenarios: list[dict]) -> dict:
    n = len(cenarios)
    entregas = sum(c["entregou"] for c in cenarios)
    tempos = [c["tempo_entrega"] for c in cenarios if c["entregou"] and c["tempo_entrega"] is not None]
    ic_baixo, ic_alto = _wilson(entregas, n)
    return {
        "n": n,
        "entrega": entregas / n,
        "ic_baixo": ic_baixo,
        "ic_alto": ic_alto,
        "coleta": sum(c["coletou"] for c in cenarios) / n,
        "colisoes_media": sum(c["colisoes"] for c in cenarios) / n,
        "tempo_entrega_medio": (sum(tempos) / len(tempos)) if tempos else None,
        "encalhes_media": sum(c["encalhes"] for c in cenarios) / n,
        "tempo_encalhado_medio": sum(c["tempo_encalhado"] for c in cenarios) / n,
        "termina_encalhado": sum(c["termina_encalhado"] for c in cenarios) / n,
    }


def _expandir(padroes: list[str]) -> list[str]:
    caminhos = []
    for padrao in padroes:
        if _eh_campos(padrao):
            caminhos.append(padrao)
            continue
        achados = sorted(glob.glob(padrao)) or ([padrao] if os.path.exists(padrao) else [])
        if not achados:
            print(f"[avaliar] AVISO: nada encontrado para {padrao}")
        caminhos.extend(a for a in achados if a not in caminhos)
    return caminhos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("redes", nargs="+", help="arquivos .npz ou padroes com *")
    parser.add_argument("--cenarios", type=int, default=100)
    parser.add_argument("--ambientes", nargs="+", choices=AMBIENTES, default=list(AMBIENTES))
    parser.add_argument("--semente-base", type=int, default=9000,
                        help="cenario k usa a semente base+k; o treino usa outras sementes")
    parser.add_argument("--ruido", type=float, default=0.0,
                        help="desvio do ruido gaussiano nas leituras dos raios")
    parser.add_argument("--sem-eliminacao", action="store_true",
                        help="o agente nunca e retirado por excesso de colisoes")
    parser.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    redes = _expandir(args.redes)
    if not redes:
        print("[avaliar] ERRO: nenhuma rede para avaliar")
        return 1

    tarefas = [(r, a, args.cenarios, args.semente_base, args.ruido, args.sem_eliminacao)
               for a in args.ambientes for r in redes]
    print(f"[avaliar] {len(redes)} redes x {len(args.ambientes)} ambientes x "
          f"{args.cenarios} cenarios, sementes {args.semente_base} a "
          f"{args.semente_base + args.cenarios - 1}, ruido nos raios {args.ruido}"
          f"{', sem eliminacao' if args.sem_eliminacao else ''}")
    with ProcessPoolExecutor(max_workers=args.processos) as pool:
        resultados = list(pool.map(_avaliar, tarefas))

    por_chave = {(r["caminho"], r["ambiente"]): r["cenarios"] for r in resultados}
    referencia = redes[0]
    linhas_csv = []

    for ambiente in args.ambientes:
        print(f"\n=== {ambiente} ===")
        print(f"{'rede':<22} {'entrega':>8} {'IC 95%':>16} {'coleta':>7} {'colisoes':>9} "
              f"{'t. entrega':>10}   vs {_rotulo(referencia)}")
        ref = por_chave[(referencia, ambiente)]
        for rede in redes:
            cenarios = por_chave[(rede, ambiente)]
            r = _resumo(cenarios)
            so_ref = sum(a["entregou"] and not b["entregou"] for a, b in zip(ref, cenarios))
            so_cand = sum(b["entregou"] and not a["entregou"] for a, b in zip(ref, cenarios))
            p = _mcnemar_exato(so_ref, so_cand)
            comparacao = ("referencia" if rede == referencia else
                          f"{(r['entrega'] - _resumo(ref)['entrega']) * 100:+5.1f} pp "
                          f"(so ela {so_cand}, so ref {so_ref}, p={p:.2f})")
            tempo = f"{r['tempo_entrega_medio']:.1f} s" if r["tempo_entrega_medio"] is not None else "-"
            print(f"{_rotulo(rede):<22} {r['entrega'] * 100:7.1f}% "
                  f"[{r['ic_baixo'] * 100:5.1f}, {r['ic_alto'] * 100:5.1f}] "
                  f"{r['coleta'] * 100:6.1f}% {r['colisoes_media']:9.1f} {tempo:>10}   {comparacao}")
            print(f"{'':<22} encalhes {r['encalhes_media']:.2f} por cenario, "
                  f"{r['tempo_encalhado_medio']:.1f} s encalhado, "
                  f"termina encalhado em {r['termina_encalhado'] * 100:.0f}%")
            linhas_csv.append({
                "rede": _rotulo(rede), "arquivo": rede if _eh_campos(rede) else os.path.relpath(rede, _RAIZ),
                "ambiente": ambiente, "ruido": args.ruido, "sem_eliminacao": args.sem_eliminacao, **r,
                "so_rede_entregou": so_cand, "so_referencia_entregou": so_ref, "p_mcnemar": p,
            })

    pasta = os.path.join(_RAIZ, "runs", "avaliacoes")
    os.makedirs(pasta, exist_ok=True)
    saida = os.path.join(pasta, f"avaliacao_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas_csv)
    print(f"\n[avaliar] tabela gravada em {os.path.relpath(saida, _RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
