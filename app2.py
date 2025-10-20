import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

# Textos streamlit
st.set_page_config(page_title="Gerador de Horários Escolares", layout="centered")
st.title("Timetabling")

# Leitura dos dados por CSV
professores = pd.read_csv("files/professores.csv")
turmas = pd.read_csv("files/turmas.csv")
disciplinas = pd.read_csv("files/disciplinas.csv")
horarios = pd.read_csv("files/horarios.csv")

# Cria o modelo CP
model = cp_model.CpModel()
# Limpas as colunas de espaços em branco
professores.columns = professores.columns.str.strip()
turmas.columns = turmas.columns.str.strip()
disciplinas.columns = disciplinas.columns.str.strip()
horarios.columns = horarios.columns.str.strip()

# Indexa professor, turma, horário e disciplina em uma MultiIndex (Matriz de 4 dimensões)
index = pd.MultiIndex.from_product(
        [professores['nome'], turmas['nome'], horarios['dia'], horarios['horario'], disciplinas['nome']],
        names=["Professor", "Turma", "Dia", "Horário", "Disciplina"]
    )
x_df = pd.DataFrame(index=index)
# Filtra professores qualificados para cada disciplina
qualificacoes_dict = {row['nome']: row['qualificações'].split(';') for _, row in professores.iterrows()}
    
mask = [False]*len(index)
for i, (p, t, di, h, d) in enumerate(index):
    mask[i] = d in qualificacoes_dict[p]
# Cria uma cópia do DataFrame filtrado e já adiciona as colunas
x_df_filtrado = x_df[mask].copy()
x_df_filtrado = x_df_filtrado.reset_index()  # transforma MultiIndex em colunas normais
x_df_filtrado["Variavel"] = x_df_filtrado.apply(
    lambda row: f"x_{row['Professor']}_{row['Turma']}_{row['Dia']}_{row['Horário'].replace(' ', '_')}_{row['Disciplina']}", axis=1)
x_df_filtrado["Obj"] = None

st.dataframe(x_df_filtrado)

if st.button("Gerar"):
    model = cp_model.CpModel()

    # Criação das variáveis e restrição de qualificação
    # Criação de variáveis diretamente usando itertuples
    for row in x_df_filtrado.itertuples():
        var = model.NewBoolVar(row.Variavel)
        x_df_filtrado.at[row.Index, "Obj"] = var


    # --- Restrições ---

    # Professores
    for (prof, dia, horario), grupo_df in x_df_filtrado.groupby(["Professor", "Dia", "Horário"]):
        model.Add(sum(grupo_df["Obj"]) <= 1)

    # Turmas
    for (turma, dia, horario), grupo_df in x_df_filtrado.groupby(["Turma", "Dia", "Horário"]):
        model.Add(sum(grupo_df["Obj"]) <= 1)

    
    # Cada disciplina deve ser atribuída exatamente uma vez por turma
    disciplinas_dict = pd.Series(disciplinas.número_de_aulas_por_semana.values,index=disciplinas.nome).to_dict()

    for (turma, disciplina), grupo_df in x_df_filtrado.groupby(["Turma", "Disciplina"]):
        model.Add(sum(grupo_df["Obj"]) == disciplinas_dict[disciplina])


    # Soma de aulas por professor usando groupby
    aulas_prof = {}
    for prof, grupo_df in x_df_filtrado.groupby("Professor"):
        lista_vars = grupo_df["Obj"].tolist()
        aulas_prof[prof] = model.NewIntVar(0, len(lista_vars), f"aulas_{prof}")
        model.Add(aulas_prof[prof] == sum(lista_vars))

    # Variáveis auxiliares
    max_aulas = model.NewIntVar(0, len(x_df_filtrado), "max_aulas")
    min_aulas = model.NewIntVar(0, len(x_df_filtrado), "min_aulas")

    for prof in aulas_prof:
        model.Add(aulas_prof[prof] <= max_aulas)
        model.Add(aulas_prof[prof] >= min_aulas)

    # Objetivo
    model.Minimize(max_aulas - min_aulas)

    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 50
    status = solver.Solve(model)

    # Exibir resultados
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        valores = []
        dados = []
        for row in x_df_filtrado.itertuples():
            val = solver.Value(row.Obj)
            valores.append(val)
            if val == 1:
                dados.append({
                    "Professor": row.Professor,
                    "Turma": row.Turma,
                    "Dia": row.Dia,
                    "Horário": row.Horário,
                    "Disciplina": row.Disciplina
                })
        
        df_result = pd.DataFrame(dados)
        
        dias_ordenados = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
        df_result["Dia"] = pd.Categorical(df_result["Dia"], categories=dias_ordenados, ordered=True)
        horarios_ordenados = ["7h", "8h", "9h", "10h", "11h"]
        df_result["Horário"] = pd.Categorical(df_result["Horário"], categories=horarios_ordenados, ordered=True)

        if df_result.empty:
            st.info("Nenhuma aula atribuída.")
        else:
            st.success("✅ Horário gerado com sucesso!")
            st.dataframe(df_result)
            
            for turma, df_turma in df_result.groupby("Turma"):
                st.subheader(f"📝 Horário da turma {turma}")
                df_pivot = df_turma.pivot_table(
                    index="Horário",
                    columns="Dia",
                    values=["Disciplina", "Professor"],
                    observed=False,
                    aggfunc=lambda x: " / ".join(x)  # caso haja múltiplos (não deve acontecer)
                )
                # Ordenar linhas (horário) e colunas (dias)
                df_pivot = df_pivot.sort_index()
                df_pivot = df_pivot.reindex(columns=dias_ordenados, level=1)  # nível 1 é 'Dia'
                df_pivot["Disciplina"] = df_pivot["Disciplina"].fillna("Sem aula")
                df_pivot["Professor"] = df_pivot["Professor"].fillna(" - ")
                df_pivot_combined = df_pivot["Disciplina"].astype(str) + " (" + df_pivot["Professor"].astype(str) + ")"

                # Mostrar tabela
                st.dataframe(df_pivot_combined)
    else:
        st.warning("Nenhuma solução ótima encontrada.")
    