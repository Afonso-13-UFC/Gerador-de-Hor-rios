import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import io


# XXXX Part I - Memoria do site XXXX

st.set_page_config(page_title="Gerador de Horários Pro", layout="wide")

# Memoria dos dias da semana
if 'config_dias' not in st.session_state:
    st.session_state['config_dias'] = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]

# Memoria de Maximo de aulas por dia
if 'config_aulas' not in st.session_state:
    st.session_state['config_aulas'] = 15 

# Memoria dos periodos de descanso
if 'horarios_bloqueados' not in st.session_state:
    st.session_state['horarios_bloqueados'] = [3, 7, 11] 
    
# Memoria de aulas cosecutivas 
if 'config_max_consecutivas' not in st.session_state:
    st.session_state['config_max_consecutivas'] = 2 

# Memoria da configuração de preferencia
if 'config_preferencia' not in st.session_state:
    st.session_state['config_preferencia'] = "Aulas Seguidas (Blocos)"

# Memoria do banco de professores
if 'banco_professores' not in st.session_state:
    st.session_state['banco_professores'] = {} 

# Memoria da Grade
if 'banco_grade' not in st.session_state:
    st.session_state['banco_grade'] = [] 

# Memoria do horario oficial
if 'horario_oficial' not in st.session_state:
    st.session_state['horario_oficial'] = None 

# XXXX Part II - Classes e Cérebro XXXX

class Professor:
    def __init__(self, nome, total_dias, total_aulas, bloqueios_escola):
        self.nome = nome
        self.horarios_disponiveis = [
            (d, h) for d in range(total_dias) for h in range(total_aulas) 
            if h not in bloqueios_escola
        ]
    def remover_disponibilidade(self, dia, horario):
        if (dia, horario) in self.horarios_disponiveis:
            self.horarios_disponiveis.remove((dia, horario))

class Turma:
    def __init__(self, nome):
        self.nome = nome

class NecessidadeAula:
    def __init__(self, turma, professor, disciplina, qtd_aulas_semana, max_aulas_dia):
        self.turma = turma
        self.professor = professor
        self.disciplina = disciplina
        self.qtd_aulas_semana = qtd_aulas_semana
        self.max_aulas_dia = max_aulas_dia

def gerar_horario(grade, lista_professores, lista_turmas, nomes_dias, num_aulas_dia, bloqueios, max_consec, preferencia):
    DIAS_SEMANA = len(nomes_dias)
    AULAS_POR_DIA = num_aulas_dia
    modelo = cp_model.CpModel()
    alocacoes = {}
    
    # Cria as variáveis principais
    for id_aula, n in enumerate(grade):
        for d in range(DIAS_SEMANA):
            for h in range(AULAS_POR_DIA):
                alocacoes[(id_aula, d, h)] = modelo.NewBoolVar(f'a_{id_aula}_d{d}_h{h}')
                if h in bloqueios: 
                    modelo.Add(alocacoes[(id_aula, d, h)] == 0)

    # Regras básicas (Quantidade na semana, Choques, Indisponibilidades)
    for id_aula, n in enumerate(grade):
        modelo.Add(sum(alocacoes[(id_aula, d, h)] for d in range(DIAS_SEMANA) for h in range(AULAS_POR_DIA)) == n.qtd_aulas_semana)
        for d in range(DIAS_SEMANA):
            for h in range(AULAS_POR_DIA):
                if (d, h) not in n.professor.horarios_disponiveis:
                    modelo.Add(alocacoes[(id_aula, d, h)] == 0)
                    
    for d in range(DIAS_SEMANA):
        for h in range(AULAS_POR_DIA):
            for prof in lista_professores:
                modelo.AddAtMostOne([alocacoes[(id_aula, d, h)] for id_aula, n in enumerate(grade) if n.professor == prof])
            for turma in lista_turmas:
                modelo.AddAtMostOne([alocacoes[(id_aula, d, h)] for id_aula, n in enumerate(grade) if n.turma == turma])
                
    # Regras de Espaçamento e Limites Diários
    consecutivas_vars = []
    for id_aula, n in enumerate(grade):
        for d in range(DIAS_SEMANA):
            # 1. Limite total daquela matéria no dia
            modelo.Add(sum(alocacoes[(id_aula, d, h)] for h in range(AULAS_POR_DIA)) <= n.max_aulas_dia)
            
            # 2. Limite restrito de aulas SEGUIDAS
            # Proíbe criar um bloco maior que o máximo permitido (ex: se max_consec=2, bloqueia 3 seguidas)
            if max_consec < AULAS_POR_DIA:
                for h in range(AULAS_POR_DIA - max_consec):
                    # Exige que em qualquer janela de tamanho (max_consec + 1), pelo menos 1 aula seja falsa (vazia/outra)
                    janela_excesso = [alocacoes[(id_aula, d, h + i)].Not() for i in range(max_consec + 1)]
                    modelo.AddBoolOr(janela_excesso)
            
            # 3. Lógica para a Inteligência Artificial Otimizar (A preferência da escola)
            for h in range(AULAS_POR_DIA - 1):
                if h in bloqueios or (h+1) in bloqueios: continue
                # Criamos um rastreador: É 1 apenas se houver aula na hora H e na hora H+1
                b = modelo.NewBoolVar(f'cons_{id_aula}_{d}_{h}')
                modelo.AddBoolAnd([alocacoes[(id_aula, d, h)], alocacoes[(id_aula, d, h+1)]]).OnlyEnforceIf(b)
                modelo.AddBoolOr([alocacoes[(id_aula, d, h)].Not(), alocacoes[(id_aula, d, h+1)].Not()]).OnlyEnforceIf(b.Not())
                consecutivas_vars.append(b)

    # Diz para a IA qual é o objetivo de vida dela (Maximizar ou Minimizar o agrupamento)
    if preferencia == "Aulas Seguidas (Blocos)":
        modelo.Maximize(sum(consecutivas_vars))
    else:
        modelo.Minimize(sum(consecutivas_vars))

    solver = cp_model.CpSolver()
    # Dá um tempo limite de segurança para a IA não ficar calculando infinitamente a otimização
    solver.parameters.max_time_in_seconds = 15.0 
    status = solver.Solve(modelo)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res = []
        for d in range(DIAS_SEMANA):
            for h in range(AULAS_POR_DIA):
                if h in bloqueios: continue
                for id_aula, n in enumerate(grade):
                    if solver.Value(alocacoes[(id_aula, d, h)]) == 1:
                        res.append({"Turma": n.turma.nome, "Dia": nomes_dias[d], "Horário": f"{h+1}º Horário", "Disciplina": n.disciplina, "Professor": n.professor.nome})
        return res
    return None

# XXXX Part II - Classes e Cérebro XXXX

st.sidebar.title("🏫 Gestão Escolar")
pag = st.sidebar.radio("Navegação", ["⚙️ Configuração da Escola", "👩‍🏫 Portal do Professor", "📅 Gerar Horários"])

# --- TELA 0: CONFIGURAÇÃO ---
if pag == "⚙️ Configuração da Escola":
    st.title("Parâmetros do Sistema")
    with st.form("config_form"):
        st.subheader("1. Estrutura de Dias e Horários")
        
        # O Multiselect permite adicionar/remover dias livremente
        dias_opcoes = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        dias_selecionados = st.multiselect("Dias Letivos da Escola:", dias_opcoes, default=st.session_state['config_dias'])
        
        n_aulas = st.number_input("Total de slots (aulas + descansos) no dia", value=st.session_state['config_aulas'])
        st.write("Selecione os slots de Intervalo/Almoço (ficarão bloqueados):")
        cols = st.columns(5)
        bloqueios_selecionados = []
        for i in range(n_aulas):
            tipo = "Aula"
            if i == 3: tipo = "Intervalo 1"
            elif i == 7: tipo = "ALMOÇO"
            elif i == 11: tipo = "Intervalo 2"
            
            if cols[i % 5].checkbox(f"{i+1}º ({tipo})", value=(i in st.session_state['horarios_bloqueados'])):
                bloqueios_selecionados.append(i)
                
        st.divider()
        st.subheader("2. Comportamento e Preferências")
        col_A, col_B = st.columns(2)
        with col_A:
            max_consec = st.number_input("Máximo de aulas SEGUIDAS permitidas por matéria", min_value=1, max_value=5, value=st.session_state['config_max_consecutivas'])
        with col_B:
            pref = st.radio("Como o sistema deve agrupar as aulas?", ["Aulas Seguidas (Blocos)", "Aulas Separadas (Gaps)"])
        
        if st.form_submit_button("Salvar Todas as Configurações"):
            if len(dias_selecionados) == 0:
                st.error("Você precisa selecionar pelo menos um dia letivo!")
            else:
                st.session_state['config_dias'] = dias_selecionados
                st.session_state['config_aulas'] = n_aulas
                st.session_state['horarios_bloqueados'] = bloqueios_selecionados
                st.session_state['config_max_consecutivas'] = max_consec
                st.session_state['config_preferencia'] = pref
                st.session_state['banco_professores'] = {} 
                st.success("Configuração salva com sucesso! O sistema recomeçará com as novas regras.")

# --- TELA 1: PROFESSOR ---
elif pag == "👩‍🏫 Portal do Professor":
    st.title("Portal do Professor")
    aba1, aba2 = st.tabs(["Minha Disponibilidade", "Meu Horário"])
    
    with aba1:
        nome = st.text_input("Seu Nome")
        dias = st.session_state['config_dias']
        aulas = st.session_state['config_aulas']
        bloqueios = st.session_state['horarios_bloqueados']
        
        nomes_linhas = []
        for i in range(aulas):
            txt = f"{i+1}º Horário"
            if i in bloqueios: txt += " (BLOQUEADO)"
            nomes_linhas.append(txt)
            
        df_disp = pd.DataFrame(True, index=nomes_linhas, columns=dias)
        for b in bloqueios: df_disp.iloc[b] = False
            
        st.info("Desmarque os horários em que você NÃO pode trabalhar.")
        edt = st.data_editor(df_disp, use_container_width=True)
        
        if st.button("Salvar Cadastro"):
            if nome == "": st.warning("Por favor, digite seu nome.")
            else:
                p = Professor(nome, len(dias), aulas, bloqueios)
                for d_idx, d_nome in enumerate(dias):
                    for h_idx in range(aulas):
                        if edt.iloc[h_idx][d_nome] == False:
                            p.remover_disponibilidade(d_idx, h_idx)
                st.session_state['banco_professores'][nome] = p
                st.success(f"Professor {nome} cadastrado com sucesso!")

    with aba2:
        st.subheader("📅 Minha Agenda")
        if st.session_state['horario_oficial'] is None:
            st.info("O horário oficial ainda não foi gerado.")
        else:
            lista_profs_com_aula = st.session_state['horario_oficial']["Professor"].unique()
            prof_logado = st.selectbox("Selecione seu nome:", lista_profs_com_aula)
            
            if prof_logado:
                agenda_prof = st.session_state['horario_oficial'][st.session_state['horario_oficial']["Professor"] == prof_logado]
                # Ordena os dias corretamente de acordo com a configuração da escola
                ordem_dias = st.session_state['config_dias']
                agenda_prof['Dia'] = pd.Categorical(agenda_prof['Dia'], categories=ordem_dias, ordered=True)
                agenda_prof = agenda_prof.sort_values(by=["Dia", "Horário"])
                st.table(agenda_prof[["Dia", "Horário", "Turma", "Disciplina"]])

# --- TELA 2: COORDENAÇÃO ---
elif pag == "📅 Gerar Horários":
    st.title("Coordenação e Grade")
    with st.expander("➕ Adicionar Aula"):
        c1, c2, c3, c4 = st.columns(4)
        t_in = c1.text_input("Turma")
        d_in = c2.text_input("Disciplina")
        p_in = c3.selectbox("Professor", list(st.session_state['banco_professores'].keys()) if len(st.session_state['banco_professores']) > 0 else ["Nenhum"])
        q_in = c4.number_input("Aulas/Semana", 1, 15, 2)
        if st.button("Inserir na Grade"):
            if p_in == "Nenhum":
                st.error("Cadastre professores primeiro!")
            else:
                st.session_state['banco_grade'].append({"Turma": t_in, "Disciplina": d_in, "Professor": p_in, "Aulas/Semana": q_in, "Max/Dia": st.session_state['config_max_consecutivas'] + 1}) # Max/dia base
                st.rerun()

    if len(st.session_state['banco_grade']) > 0:
        st.write("### Grade Solicitada")
        st.dataframe(pd.DataFrame(st.session_state['banco_grade']), use_container_width=True)
        if st.button("🗑️ Limpar Tudo"):
            st.session_state['banco_grade'] = []
            st.rerun()

    if st.button("🚀 Gerar Horário Oficial", type="primary"):
        turmas_dict = {}
        grade_ia = []
        for item in st.session_state['banco_grade']:
            if item["Turma"] not in turmas_dict: turmas_dict[item["Turma"]] = Turma(item["Turma"])
            prof_obj = st.session_state['banco_professores'][item["Professor"]]
            grade_ia.append(NecessidadeAula(turmas_dict[item["Turma"]], prof_obj, item["Disciplina"], item["Aulas/Semana"], item["Max/Dia"]))
        
        with st.spinner("A IA está buscando a distribuição perfeita... (Isso pode levar alguns segundos)"):
            res = gerar_horario(
                grade=grade_ia, 
                lista_professores=list(st.session_state['banco_professores'].values()), 
                lista_turmas=list(turmas_dict.values()), 
                nomes_dias=st.session_state['config_dias'], 
                num_aulas_dia=st.session_state['config_aulas'], 
                bloqueios=st.session_state['horarios_bloqueados'],
                max_consec=st.session_state['config_max_consecutivas'],
                preferencia=st.session_state['config_preferencia']
            )
        
        if res:
            st.success("🎉 Sucesso! Horário gerado com o melhor agrupamento possível.")
            df_res = pd.DataFrame(res)
            st.session_state['horario_oficial'] = df_res
            st.dataframe(df_res, use_container_width=True)
            
            st.divider()
            st.subheader("Opções de Exportação")
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                # 1. Exportação em Lista para planilhas
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Baixar Lista Padrão (Excel/CSV)",
                    data=csv,
                    file_name='horario_escola.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
            with col_btn2:
                # 2. FORMATADA PARA PDF
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Horário Escolar</title>
                    <style>
                        /* Ajustes gerais de tamanho para caber no A4 */
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ text-align: center; color: #333; font-size: 22px; margin-bottom: 5px; }}
                        h2 {{ color: #444; margin-top: 15px; border-bottom: 2px solid #333; padding-bottom: 5px; font-size: 18px; }}
                        
                        /* O table-layout: fixed impede que a tabela "vaze" para o lado */
                        table {{ width: 100%; table-layout: fixed; border-collapse: collapse; margin-top: 10px; text-align: center; page-break-inside: avoid; }}
                        
                        /* Reduzimos o padding (espaço interno) de 10px para 4px */
                        th, td {{ border: 1px solid #000; padding: 4px 2px; word-wrap: break-word; }} 
                        
                        th {{ background-color: #f4f4f4; font-weight: bold; font-size: 12px; }}
                        .bloqueio {{ background-color: #e9ecef; font-weight: bold; color: #555; letter-spacing: 1px; font-size: 11px; }}
                        
                        /* Fontes ligeiramente menores para encaixar perfeitamente */
                        .disciplina {{ font-weight: bold; font-size: 12px; }} 
                        .professor {{ font-size: 10px; color: #555; }} 
                        
                        /* Regras específicas para a impressora */
                        @media print {{
                            @page {{ size: A4 portrait; margin: 10mm; }} /* Força o formato A4 retrato e margens finas */
                            body {{ margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }} /* Garante que o cinza do intervalo apareça na impressão */
                            .page-break {{ page-break-before: always; }}
                        }}
                    </style>
                </head>
                <body onload="setTimeout(() => window.print(), 500)">
                    <h1>🏫 Grade Oficial de Horários</h1>
                """

                # Cria uma grade nova para cada Turma
                for idx, turma in enumerate(df_res['Turma'].unique()):
                    if idx > 0:
                        html_content += "<div class='page-break'></div>" # Quebra de página
                        
                    html_content += f"<h2>Turma: {turma}</h2>"
                    html_content += "<table>"
                    
                    # Cabeçalho: Horário + Dias da Semana
                    html_content += "<tr><th>Horário</th>"
                    for d in st.session_state['config_dias']:
                        html_content += f"<th>{d}</th>"
                    html_content += "</tr>"
                    
                    # Linhas: Os slots de aula
                    df_turma = df_res[df_res['Turma'] == turma]
                    for h in range(st.session_state['config_aulas']):
                        html_content += "<tr>"
                        html_content += f"<td><b>{h+1}º Horário</b></td>"
                        
                        if h in st.session_state['horarios_bloqueados']:
                            html_content += f"<td colspan='{len(st.session_state['config_dias'])}' class='bloqueio'>INTERVALO / ALMOÇO</td>"
                        else:
                            for d in st.session_state['config_dias']:
                                aula = df_turma[(df_turma['Dia'] == d) & (df_turma['Horário'] == f"{h+1}º Horário")]
                                if not aula.empty:
                                    disc = aula.iloc[0]['Disciplina']
                                    prof = aula.iloc[0]['Professor']
                                    html_content += f"<td><div class='disciplina'>{disc}</div><div class='professor'>{prof}</div></td>"
                                else:
                                    html_content += "<td>-</td>" # Janela vaga
                        html_content += "</tr>"
                        
                    html_content += "</table>"
                html_content += "</body></html>"
                
                st.download_button(
                    label="🖨️ Gerar PDF Formatado (Grade)",
                    data=html_content.encode('utf-8'),
                    file_name='Grade_Escolar.html',
                    mime='text/html',
                    use_container_width=True
                )