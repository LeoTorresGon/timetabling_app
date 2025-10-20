import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

st.set_page_config(page_title="Gerador de Horários Escolares", layout="centered")
st.title("📅 Gerador de Horários Escolares — Versão Corrigida")

dados = {
    'professores': ["Ana", "Bruno", "Carla", "Daniel"],
    'turmas': ["1A", "1B", "2A"],
    'horarios': ["Segunda 8h", "Segunda 10h", "Terça 8h", "Terça 10h"],
    'disciplinas': ["Matemática", "Português", "Ciências"]
}

# --- Entradas ---
professores = st.multiselect(
    "Professores disponíveis:",
    dados['professores'],
    default=["Ana", "Bruno"]
)

turmas = st.multiselect(
    "Turmas:",
    dados['turmas'],
    default=["1A", "1B"]
)

horarios = st.multiselect(
    "Horários disponíveis:",
    dados['horarios'],
    default=["Segunda 8h", "Terça 8h"]
)

disciplinas = st.multiselect(
    "Disciplinas disponíveis:",
    dados['disciplinas'],
    default=["Matemática", "Português", "Ciências"]
)

if professores and turmas and horarios and disciplinas:
    index = pd.MultiIndex.from_product(
        [professores, turmas, horarios, disciplinas],
        names=["Professor", "Turma", "Horário", "Disciplina"]
    )
    x_df = pd.DataFrame(index=index)
    x_df["Variável"] = [f"x_{p}_{t}_{h.replace(' ', '_')}_{d}" for p, t, h, d in index]

    st.subheader("📊 Variáveis possíveis")
    st.dataframe(x_df)

if st.button("🔧 Gerar Horário"):
    if not professores or not turmas or not horarios:
        st.warning("Preencha todas as listas antes de gerar.")
    else:
        model = cp_model.CpModel()

        # Criação das variáveis
        for p, t, h, d in x_df.index:
            x_df.loc[(p, t, h, d), "Obj"] = model.NewBoolVar(x_df.loc[(p, t, h, d), "Variável"])

        # --- Restrições ---

        # 1. Cada turma deve ter pelo menos 1 aula no total
        for t in turmas:
            model.Add(sum(x_df.loc[(p, t, h, d), "Obj"] for p in professores for h in horarios) >= 1)

        # 2. Turma não pode ter duas aulas no mesmo horário
        for t in turmas:
            for h in horarios:
                model.Add(sum(x_df.loc[(p, t, h), "Obj"] for p in professores) <= 1)

        # 3. Professor não pode dar duas aulas ao mesmo tempo
        for p in professores:
            for h in horarios:
                model.Add(sum(x_df.loc[(p, t, h), "Obj"] for t in turmas) <= 1)

        # Objetivo: equilibrar a carga de aulas entre professores
        model.Maximize(sum(x_df["Obj"]))

        # Resolver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)

        # Exibir resultados
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            valores = []
            dados = []
            for (p, t, h), row in x_df.iterrows():
                val = solver.Value(row["Obj"])
                valores.append(val)
                if val == 1:
                    dados.append({"Professor": p, "Turma": t, "Horário": h})
            x_df["Valor"] = valores

            st.subheader("📈 Valores das variáveis (0 = inativo, 1 = ativo)")
            st.dataframe(x_df)

            df_result = pd.DataFrame(dados)
            if df_result.empty:
                st.info("Nenhuma aula atribuída.")
            else:
                st.success("✅ Horário gerado com sucesso!")
                st.dataframe(df_result)
        else:
            st.error("❌ Nenhuma solução encontrada.")
