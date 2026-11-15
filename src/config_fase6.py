from simulador import CenarioConfig

# Fase 6: topologia de corredores, com paredes longas e aberturas.
#
# Contornar uma parede afasta o agente do alvo por um tempo, entao os termos
# de aproximacao e alinhamento foram reduzidos, a eliminacao por colisao ficou
# mais tolerante e ficar preso passou a custar por segundo.

CONFIG = CenarioConfig(
    nome="Fase 6 - corredores de armazem",
    checkpoint_entrada="melhor_rede_fase5.npz",
    checkpoint_saida="melhor_rede_fase6.npz",

    num_agentes=30,
    duracao_geracao=55.0,
    num_geracoes=250,
    elite=4,

    # Mutacao menor: a rede da fase 5 e refinada, nao explorada do zero.
    taxa_mutacao=0.12,
    forca_mutacao=0.15,

    usa_obstaculos=True,
    usa_layout_corredores=True,
    num_fileiras=2,
    espessura_fileira=60,
    largura_corredor_min=110,
    num_transversais=2,
    largura_transversal=120,

    num_robos_moveis=3,
    # Layouts fixos, os mesmos em todas as geracoes.
    cenarios_por_geracao=6,
    layouts_fixos=6,
    seed=42,

    peso_progresso=2.0,
    peso_alinhamento=0.01,
    # Medido pela melhor aproximacao alcancada, nao pela posicao final.
    bonus_proximidade_max=800.0,
    proximidade_por_melhor_aproximacao=True,
    penalidade_colisao=300.0,
    penalidade_colisao_continua=60.0,
    max_colisoes_morte=12,
)
