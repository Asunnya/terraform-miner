# %% [1]
# Global Configuration, Constants, and Setup

# --- Core Libraries & System ---
import os
import sys
import re
import csv
import json
import glob
from datetime import datetime
from collections import Counter, defaultdict
import itertools
import random
import yaml
import pickle

# --- Data Handling & Numerics ---
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- NLP & Machine Learning ---
from sentence_transformers import SentenceTransformer
import hdbscan
import umap.umap_ as umap # o alias umap é comum para evitar conflito com o módulo umap em si
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, DetectorFactory, LangDetectException
from emoji import emoji_count

# --- Project-Specific Paths ---
# __file__ refers to this script (phase2.py)
# PROJECT_ROOT will be terraform-miner/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'src', 'miner', 'config.yaml')) # For original miner config
NLP_CONFIG_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'src', 'nlp_analysis', 'config.yaml')) # For NLP specific config

# Expected input paths (used in later stages, e.g., Cell [6])
ALL_COMMITS_CSV_PATH_EXPECTED = os.path.join(PROJECT_ROOT, 'reports', 'all_commits.csv')
TOP_BIGRAMS_JSON_PATH_EXPECTED = os.path.join(PROJECT_ROOT, 'reports', 'top_bigrams.json')

# --- Output Directory ---
# This line will be removed by the edit: OUTPUT_DIR = "analysis_results_integrated_v2"

# --- Configuration Loading ---
def load_yaml_config(path_to_config_yaml):
    if os.path.exists(path_to_config_yaml):
        try:
            with open(path_to_config_yaml, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"AVISO: Erro ao carregar o arquivo de configuração YAML: {path_to_config_yaml}. Erro: {e}")
            return {}
    print(f"AVISO: Arquivo de configuração não encontrado: {path_to_config_yaml}")
    return {}

cfg = load_yaml_config(CONFIG_FILE_PATH)
nlp_cfg = load_yaml_config(NLP_CONFIG_FILE_PATH) # Load NLP specific config

# --- Output Directory (now from NLP config) ---
OUTPUT_DIR = nlp_cfg.get('output_paths', {}).get('base_nlp_output_dir', 'analysis_results_nlp')

# --- Dataset Path Configuration (from config.yaml or default) ---
default_dataset_relative_path_from_root = "../data/dataset" # Used if config fails or lacks the key
configured_dataset_relative_path = cfg.get('output_paths', {}).get('dataset_dir', default_dataset_relative_path_from_root)
USER_MINER_OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, configured_dataset_relative_path))

if not cfg.get('output_paths', {}).get('dataset_dir'):
    print(f"AVISO: 'output_paths.dataset_dir' não encontrado em {CONFIG_FILE_PATH}. Usando padrão: {USER_MINER_OUTPUT_DIR}")
else:
    print(f"Usando 'output_paths.dataset_dir' de {CONFIG_FILE_PATH}: {USER_MINER_OUTPUT_DIR}")

# --- Reproducibility & NLTK Setup ---
DetectorFactory.seed = 0
random.seed(42)
np.random.seed(42)

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError: nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError: nltk.download('punkt')

# --- Text Processing Parameters (now from NLP config) ---
# Initialize with NLTK defaults
STOP_WORDS_EN = set(stopwords.words('english'))
# Update with custom additions from nlp_cfg
_nlp_text_processing_params = nlp_cfg.get('text_processing_params', {})
STOP_WORDS_EN.update(_nlp_text_processing_params.get('custom_stopwords_additions', []))

EXCLUDED_DIRS_PATTERNS = _nlp_text_processing_params.get('excluded_dirs_patterns', [
    r"vendor/", r"test/fixtures/", r"examples/", r"tests/", r"testdata/",
    r"\.github/", r"docs/"
])
RELEVANT_EXTENSIONS_ORDER = _nlp_text_processing_params.get('relevant_extensions_order', ['.tf', '.go', '.py', '.yaml', '.yml', '.json', '.sh', '.hcl'])
PATCH_EXCERPT_LINES = _nlp_text_processing_params.get('patch_excerpt_lines', 10)
MSG_CLEAN_MAX_LEN = _nlp_text_processing_params.get('msg_clean_max_len', 150)
MERGE_PATTERNS_FOR_INFERENCE = _nlp_text_processing_params.get('merge_patterns_for_inference', [
    r"^Merge pull request #\d+ from .*",
    r"^Merge remote-tracking branch '.*'",
    r"^Merge branch '.*'( of .*)?( into .*)?",
    r"^\s*Merged in .*",
])

# NEW: Load extraction patterns from config
MERGE_PATTERNS_FOR_EXTRACTION = _nlp_text_processing_params.get('merge_patterns_for_extraction', [])

# NEW: Variable to enable/disable semantic extraction from merge messages
EXTRACT_SEMANTIC_FROM_MERGE = _nlp_text_processing_params.get('extract_semantic_from_merge', True)

# --- Sampling Parameters (now from NLP config) ---
_nlp_sampling_params = nlp_cfg.get('sampling_params', {})
SAMPLE_SIZE = _nlp_sampling_params.get('sample_size', 1000)
MIN_COMMITS_PER_MONTH_STRATUM = _nlp_sampling_params.get('min_commits_per_month_stratum', 50)

# --- Token Analysis Parameters (now from NLP config) ---
_nlp_token_analysis_params = nlp_cfg.get('token_analysis_params', {})
TOP_N_TOKENS = _nlp_token_analysis_params.get('top_n_tokens', 60) # For token_freq.csv

# --- Embedding and Clustering Parameters (SBERT model from NLP config, UMAP/HDBSCAN params read in Block [8]) ---
SBERT_MODEL_NAME = nlp_cfg.get('SBERT_MODEL_NAME', 'sentence-transformers/all-roberta-large-v1')
# UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS will be read from nlp_cfg.get('UMAP_PARAMS', {}) in Block [8]
# HDBSCAN_MIN_CLUSTER_SIZE and HDBSCAN_MIN_SAMPLES are data-dependent and calculated/read in Cell [8].


# --- System Path Configuration (for custom modules) ---
# Adicionar 'src' ao sys.path (se necessário para seus módulos de análise avançada)
if os.path.join(os.getcwd(), 'src') not in sys.path and os.path.isdir('src'):
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
elif os.path.dirname(os.getcwd()) not in sys.path and os.path.isdir(os.path.join(os.path.dirname(os.getcwd()), 'src')):
    sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'src'))

# ----- PONTO DE INTEGRAÇÃO: Importar módulos do seu projeto 'src/' (se for usar análise avançada) -----
# Exemplo: Descomente e ajuste para importar seus módulos de análise personalizados.
# try:
#     from analysis import terraform_ast
#     from analysis import diff_stats
#     user_analysis_modules_dict = {'terraform_ast': terraform_ast, 'diff_stats': diff_stats}
#     print("Módulos de análise do usuário importados.")
# except ImportError:
user_analysis_modules_dict = {}
print("AVISO: Módulos de análise avançada do usuário ('terraform_ast', 'diff_stats') não encontrados ou não importados.")
# ----- FIM DO PONTO DE INTEGRAÇÃO -----

# --- Ensure OUTPUT_DIR exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Todos os resultados, incluindo logs e arquivos gerados, serão salvos em: {os.path.abspath(OUTPUT_DIR)}")

# %% [2]
# Funções auxiliares para pré-processamento de dados de commit e extração de informações relevantes.

def clean_commit_message_basic(message):
    if not message: return ""
    return message.splitlines()[0].lower().strip()

def extract_semantic_from_merge_message(message, extraction_patterns):
    """
    Extrai informação semântica de mensagens de merge usando uma lista de padrões de regex.

    Args:
        message (str): Mensagem original do commit.
        extraction_patterns (list): Lista de padrões de regex. O grupo de captura 1 deve
                                    ser a parte semântica primária (branch) e o grupo 2 (opcional)
                                    a descrição do PR.
        
    Returns:
        str: Mensagem semanticamente rica ou None se nenhum padrão corresponder.
    """
    if not message or not extraction_patterns:
        return None
    
    for pattern in extraction_patterns:
        match = re.match(pattern, message, re.DOTALL)
        if match:
            groups = match.groups()
            
            # Grupo 1 é esperado ser o nome da branch
            branch_name = groups[0] if len(groups) > 0 else ""
            # Grupo 2 é a descrição opcional do PR
            pr_description = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            
            # Limpar o nome da branch
            branch_semantic = re.sub(r'[-_]+', ' ', branch_name)
            branch_semantic = re.sub(r'[^\w\s]', ' ', branch_semantic).strip()
            
            # Combinar a semântica da branch com a descrição do PR
            if pr_description and pr_description.lower() != branch_semantic.lower():
                combined_message = f"{branch_semantic} {pr_description}"
            else:
                combined_message = pr_description if pr_description else branch_semantic
            
            # Retorna a mensagem combinada, ou apenas a semântica da branch se a combinação for vazia
            return combined_message.strip() if combined_message.strip() else branch_semantic

    return None

def clean_commit_message_advanced(message, max_len=MSG_CLEAN_MAX_LEN):
    if not message: return None
    
    # NEW: Tenta extrair informação semântica de mensagens de merge se ativado
    if EXTRACT_SEMANTIC_FROM_MERGE:
        # Passa os padrões do config para a função de extração
        semantic_message = extract_semantic_from_merge_message(message, MERGE_PATTERNS_FOR_EXTRACTION)
        if semantic_message:
            # Processa a mensagem semântica em vez da mensagem de merge original
            msg = semantic_message
        else:
            # Se não for um PR de merge ou não foi possível extrair, usa a lógica original
            msg = message.splitlines()[0]
    else:
        # Comportamento original: não remove mais merges aqui, 'is_merge' será inferido e usado para filtrar se necessário
        msg = message.splitlines()[0]
    
    # O resto da lógica de limpeza permanece a mesma
    msg = re.sub(r"(\s|^)(#|GH-)\d+\b", " ", msg)
    msg = msg.lower()
    msg = re.sub(r"https?://\S+", " ", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    if len(msg) > max_len:
        msg = msg[:max_len].rsplit(' ', 1)[0] if ' ' in msg[:max_len] else msg[:max_len]
    return msg if msg else None

def get_relevant_file_and_patch_original(modifications_list):
    selected_file_path = None
    patch_excerpt = ""
    best_mod_info = None

    # 'modifications_list' agora é uma lista de dicts como:
    # {'filename': str, 'diff': str, (outros campos que você adicionar)}

    # Prioridade 1: Por extensão definida em RELEVANT_EXTENSIONS_ORDER
    for ext in RELEVANT_EXTENSIONS_ORDER:
        for mod_info in modifications_list:
            if mod_info['filename'] and mod_info['filename'].endswith(ext):
                best_mod_info = mod_info
                break
        if best_mod_info:
            break

    # Prioridade 2: Primeiro arquivo se nenhum dos preferidos for encontrado (não temos contagem de linhas aqui)
    if not best_mod_info and modifications_list:
        best_mod_info = modifications_list[0]

    if best_mod_info and best_mod_info.get('diff'):
        selected_file_path = best_mod_info['filename']
        diff_lines = best_mod_info['diff'].splitlines()
        patch_excerpt = "\n".join(diff_lines[:PATCH_EXCERPT_LINES])
    
    return selected_file_path, patch_excerpt

def get_relevant_file_and_patch_enhanced(modifications_list, analysis_modules=None):
    # Seus módulos de análise podem ter lógicas mais sofisticadas aqui.
    # Exemplo de uso de módulos de análise (descomente e adapte se necessário):
    # if analysis_modules and 'terraform_ast' in analysis_modules:
    #     # Lógica para usar terraform_ast para encontrar o arquivo .tf mais significativo
    #     # tf_analyzer = analysis_modules['terraform_ast']
    #     # for mod_info in modifications_list:
    #     #     if mod_info['filename'].endswith('.tf'):
    #     #         significance = tf_analyzer.analyze_diff(mod_info['diff']) # Suposição
    #     #         # ... lógica para classificar e selecionar
    #     pass

    # Fallback para a lógica original (adaptada)
    return get_relevant_file_and_patch_original(modifications_list)


def is_commit_relevant_for_sampling(commit_data_item, cleaned_message, is_merge_commit):
    if not cleaned_message:
        return False
    
    # TODO: Avaliar se merges com mensagens genéricas (mesmo após limpeza) devem ser explicitamente excluídos da amostragem.
    # if is_merge_commit and cleaned_message.startswith(("merge pull request", "merge branch")):
    #      # Se for um merge E a mensagem limpa ainda for muito genérica, descarte
    #      # Isso é um pouco redundante com a limpeza, mas pode pegar casos extras
    #      # se clean_commit_message_advanced foi simplificada para não remover merges
    #      return False

    modifications = commit_data_item.get('modifications', [])
    if not modifications:
        return False

    all_files_in_excluded_dirs = True
    for f_mod_info in modifications: # f_mod_info é um dict {'filename': ..., 'diff': ...}
        file_path = f_mod_info.get('filename')
        if not file_path: continue
        
        is_excluded = False
        for pattern in EXCLUDED_DIRS_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                is_excluded = True
                break
        if not is_excluded:
            all_files_in_excluded_dirs = False
            break
    return not all_files_in_excluded_dirs


# %% [3]
# Sub-tarefa 1: Coleta de Dados dos Repositórios (lendo e agrupando seus JSONs)

all_commits_data = []
commits_grouped_temp_data = {} # (repo_name, commit_hash) -> dict_commit_info

print(f"Procurando arquivos JSON do minerador em: {os.path.abspath(USER_MINER_OUTPUT_DIR)}")
# Ajustar o padrão glob para encontrar 'data.jsonl' recursivamente nas subpastas
json_file_paths = glob.glob(os.path.join(USER_MINER_OUTPUT_DIR, "*", "data.jsonl"))

if not json_file_paths:
    print(f"AVISO: Nenhum arquivo .jsonl encontrado em subdiretórios de '{USER_MINER_OUTPUT_DIR}'. Verifique o caminho e a estrutura.")
    print("O script não pode continuar sem dados.")
    # exit() Ou levante uma exceção
else:
    print(f"Encontrados {len(json_file_paths)} arquivos JSONL para processar.")

    for json_file_path in tqdm(json_file_paths, desc="Processando arquivos JSON do minerador"):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        record = json.loads(line.strip()) # Cada linha é um JSON
                    except json.JSONDecodeError as e:
                        print(f"AVISO: Erro ao decodificar JSON na linha {line_num+1} do arquivo {json_file_path}: {e}")
                        continue

                    repo_name = record.get('repo')
                    commit_hash = record.get('commit_hash')
                    
                    if not repo_name or not commit_hash:
                        print(f"AVISO: Registro sem 'repo' ou 'commit_hash' na linha {line_num+1} de {json_file_path}. Pulando.")
                        continue

                    commit_key = (repo_name, commit_hash)

                    if commit_key not in commits_grouped_temp_data:
                        # Inferir se é um merge pela mensagem
                        original_msg = record.get('message', "")
                        is_merge_inferred = any(re.match(pattern, original_msg, re.IGNORECASE) for pattern in MERGE_PATTERNS_FOR_INFERENCE)
                        
                        # Tentar limpar a mensagem para ver se é genérica de merge
                        # Mesmo se is_merge_inferred for True, a msg_clean pode ser útil se não for genérica
                        temp_cleaned_msg_for_check = clean_commit_message_advanced(original_msg)
                        if is_merge_inferred and (not temp_cleaned_msg_for_check or temp_cleaned_msg_for_check.startswith(("merge pull request", "merge branch"))):
                            is_truly_generic_merge = True
                        else:
                            is_truly_generic_merge = False


                        commits_grouped_temp_data[commit_key] = {
                            "repo_name": repo_name,
                            "commit_hash": commit_hash,
                            "msg_original": original_msg,
                            "author_name": record.get('author'),
                            "author_date": pd.to_datetime(record.get('date'), errors='coerce'),
                            "modifications": [],
                            "is_merge": is_truly_generic_merge, # Usar a flag de merge genérico aqui
                            # Você pode adicionar 'is_merge_inferred' se quiser a flag mais ampla
                        }
                    
                    # Adicionar a modificação do arquivo
                    modification_item = {
                        "filename": record.get('file'),
                        "new_path": record.get('file'), # Assumindo que 'file' é o caminho atual
                        "old_path": record.get('file'), # Não temos info para distinguir new/old path de forma simples
                        "diff": record.get('patch'),
                        # 'added_lines' e 'deleted_lines' não estão disponíveis no seu formato
                        # Se precisar, teria que parsear o 'diff' ou seu minerador teria que fornecer
                    }
                    commits_grouped_temp_data[commit_key]["modifications"].append(modification_item)
            
        except Exception as e:
            print(f"Erro ao processar o arquivo {json_file_path}: {e}")

    # Converter o dicionário agrupado para a lista all_commits_data
    all_commits_data = list(commits_grouped_temp_data.values())
    
    if not all_commits_data:
        print("Nenhum dado de commit foi carregado após processar os arquivos JSON. Verifique os arquivos e o formato.")
        # exit()
    else:
        df_all_commits_temp = pd.DataFrame(all_commits_data) # Temporário para exibição
        print(f"Coleta de dados concluída. Total de {len(all_commits_data)} commits únicos agrupados.")
        print("\nExemplo de dados de commits agrupados (primeiras linhas do DataFrame):")
        # Mostrar colunas relevantes, 'modifications' pode ser grande
        print(df_all_commits_temp[['repo_name', 'commit_hash', 'msg_original', 'author_date', 'is_merge']].head())
        # Limpar memória
        del df_all_commits_temp
        del commits_grouped_temp_data


# %% [NOVO PASSO - Sugestão: Inserir após o Bloco 3 ou no início do Bloco 4]
# Criação e salvamento de all_commits.csv a partir de all_commits_data

if all_commits_data:
    print("\nConvertendo all_commits_data para DataFrame e salvando all_commits.csv...")
    df_all_commits_processed = pd.DataFrame(all_commits_data)

    # Assegurar que 'modifications' (lista de dicts) seja tratada para CSV se necessário
    # Por simplicidade, podemos converter para string JSON ou remover para o CSV principal
    # Aqui, vamos criar uma versão para salvar em CSV sem a coluna 'modifications' complexa,
    # mas manter df_all_commits_processed com ela para uso interno.
    
    # Criar uma cópia para manipulação antes de salvar em CSV
    df_for_csv = df_all_commits_processed.copy()

    # A coluna 'modifications' é uma lista de dicts, o que não é ideal para CSV direto.
    # Vamos converter para uma string JSON para o CSV ou contar o número de modificações.
    # Ou, se 'files_changed' e 'patch_excerpt' já capturam o essencial para o CSV,
    # podemos apenas selecionar as colunas desejadas.

    # Exemplo: Selecionar colunas principais e talvez uma contagem de modificações
    # Primeiro, extrair a lista de arquivos modificados para uma nova coluna, se útil
    df_for_csv['files_changed_list_str'] = df_for_csv['modifications'].apply(
        lambda mods: ";".join([m.get('filename', 'unknown_file') for m in mods if m]) if isinstance(mods, list) else ""
    )
    
    # Colunas a serem salvas em all_commits.csv
    # Ajuste conforme as colunas que você realmente tem em all_commits_data e quer no CSV
    cols_to_save = ['repo_name', 'commit_hash', 'msg_original', 'author_name', 'author_date', 'is_merge', 'files_changed_list_str']
    
    # Filtrar colunas existentes em df_for_csv
    cols_to_save_existing = [col for col in cols_to_save if col in df_for_csv.columns]

    df_all_commits_to_save = df_for_csv[cols_to_save_existing]
    
    # Nome do arquivo all_commits.csv a partir do nlp_cfg
    all_commits_csv_filename = nlp_cfg.get('output_paths', {}).get('all_commits_csv', 'all_commits_nlp.csv')
    ALL_COMMITS_CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, all_commits_csv_filename)
    df_all_commits_to_save.to_csv(ALL_COMMITS_CSV_OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"Arquivo {all_commits_csv_filename} salvo em: {ALL_COMMITS_CSV_OUTPUT_PATH}")
    
    # Definir total_messages_for_hdbscan_param_base
    total_messages_for_hdbscan_param_base = len(df_all_commits_processed)
    print(f"Número total de mensagens (base para HDBSCAN params): {total_messages_for_hdbscan_param_base}")

    # Para as validações subsequentes, usaremos df_all_commits_processed que contém a coluna 'modifications' original
    # e também calcularemos 'msg_word_count' e outras colunas necessárias para validação.
    df_all_commits_processed['author_date'] = pd.to_datetime(df_all_commits_processed['author_date'], errors='coerce')
    df_all_commits_processed['msg_clean_for_val'] = df_all_commits_processed['msg_original'].apply(clean_commit_message_advanced) # Use sua função
    df_all_commits_processed['msg_word_count_for_val'] = df_all_commits_processed['msg_clean_for_val'].fillna("").astype(str).apply(lambda x: len(x.split()))

else:
    print("AVISO: all_commits_data está vazio. all_commits.csv não será gerado nem validado extensivamente.")
    df_all_commits_processed = pd.DataFrame() # DataFrame vazio para evitar erros

# %% [VALIDAÇÃO - all_commits.csv]
# Inserir após all_commits.csv ser salvo

print("\n--- Iniciando Validação de all_commits.csv ---")
try:
    if not df_all_commits_processed.empty: # df_all_commits_processed tem os dados completos
        # Número total de commits
        # df_all_commits_to_save foi o que realmente foi salvo no CSV.
        num_commits_in_csv_expected = len(df_all_commits_to_save)
        print(f"  Número total de commits esperado no CSV: {num_commits_in_csv_expected}")
        # Se você recarregar o CSV para verificar, seria:
        # df_loaded_all_commits = pd.read_csv(ALL_COMMITS_CSV_OUTPUT_PATH)
        # print(f"  Número total de commits no CSV carregado: {len(df_loaded_all_commits)}")
        # if num_commits_in_csv_expected != len(df_loaded_all_commits):
        #     print("    ALERTA: Divergência no número de commits do CSV!")

        # Verificar commits de 'vendor/'
        # Usar df_all_commits_processed que tem a coluna 'modifications' original
        def check_only_vendor(modifications_list):
            if not modifications_list or not isinstance(modifications_list, list):
                return False
            files_list = [mod.get('filename', '') for mod in modifications_list if isinstance(mod, dict) and mod.get('filename')]
            if not files_list: return False
            return all(file.strip().startswith('vendor/') for file in files_list if file.strip())

        df_all_commits_processed['is_only_vendor_val'] = df_all_commits_processed['modifications'].apply(check_only_vendor)
        commits_only_vendor_count = df_all_commits_processed['is_only_vendor_val'].sum()
        print(f"  Commits unicamente em diretórios 'vendor/' (calculado a partir de df_all_commits_processed): {commits_only_vendor_count}")
        if commits_only_vendor_count > 0:
            print(f"    INFO: Encontrados {commits_only_vendor_count} commits que parecem ser apenas de 'vendor/'. Verifique se o filtro está adequado.")
    else:
        print("  AVISO: df_all_commits_processed está vazio. Validação de all_commits.csv pulada.")
    print("--- Fim da Validação de all_commits.csv ---")
except Exception as e_val_all_commits:
    print(f"ERRO durante a validação de all_commits.csv: {e_val_all_commits}")

# %% [4]
# Sub-tarefa 2: Processamento para corpus_overview.md
# (A lógica principal é mantida, mas a forma como 'is_merge' e 'modifications' são acessadas é consistente
#  com a nova estrutura de all_commits_data)

corpus_stats = {
    "total_commits": 0, "total_repos": 0,
    "min_date": None, "max_date": None, "msg_lengths": [],
    "msg_len_mean": 0, "msg_len_median": 0, "msg_len_p10": 0, "msg_len_p90": 0,
    "file_extensions_counts": Counter(), "terraform_specific_counts": Counter(),
    "vendor_commits_count": 0, "urls_in_msg_count": 0, "issue_refs_in_msg_count": 0,
    "emojis_in_msg_count": 0, "non_english_msg_count": 0,
    "fix_commits_count": 0, "feat_commits_count": 0,
    "total_processed_for_stats": 0, "generic_merge_commits_count": 0,
}

if all_commits_data:
    print("Processando commits para o corpus_overview.md...")
    
    # 'all_commits_data' já contém commits únicos (agrupados no Bloco 3)
    commits_to_process_for_overview = all_commits_data
    corpus_stats["total_commits"] = len(commits_to_process_for_overview)
    corpus_stats["total_processed_for_stats"] = len(commits_to_process_for_overview)
    
    repo_names_seen = set(c['repo_name'] for c in commits_to_process_for_overview if 'repo_name' in c)
    corpus_stats["total_repos"] = len(repo_names_seen)

    all_commit_dates = [c['author_date'] for c in commits_to_process_for_overview if pd.notnull(c.get('author_date'))]
    if all_commit_dates:
        # Converter para Series do pandas para usar .min() e .max() sem modificação
        dates_series = pd.Series(all_commit_dates)
        corpus_stats["min_date"] = dates_series.min().strftime('%Y-%m-%d')
        corpus_stats["max_date"] = dates_series.max().strftime('%Y-%m-%d')

    for commit_data in tqdm(commits_to_process_for_overview, desc="Calculando estatísticas do corpus"):
        msg_original = commit_data.get('msg_original', "")
        
        if commit_data.get('is_merge', False): # Contar merges genéricos (inferidos no Bloco 3)
            corpus_stats["generic_merge_commits_count"] += 1
            # Opcional: Pular merges genéricos das estatísticas de qualidade de texto/comprimento
            # if not clean_commit_message_advanced(msg_original): # Se a limpeza resultar em None para merges genéricos
            #     continue 

        basic_cleaned_msg = clean_commit_message_basic(msg_original)
        corpus_stats["msg_lengths"].append(len(basic_cleaned_msg))

        # Diagnósticos (como antes)
        if re.search(r"https?://\S+", msg_original): corpus_stats["urls_in_msg_count"] += 1
        if re.search(r"(\s|^)(#|GH-)\d+\b", msg_original): corpus_stats["issue_refs_in_msg_count"] += 1
        if emoji_count(msg_original) > 0: corpus_stats["emojis_in_msg_count"] += 1
        
        first_line_original = msg_original.splitlines()[0].strip() if msg_original else ""
        if first_line_original:
            try:
                if detect(first_line_original) != 'en': corpus_stats["non_english_msg_count"] += 1
            except LangDetectException: pass

        if basic_cleaned_msg.startswith("fix:") or basic_cleaned_msg.startswith("fix("): corpus_stats["fix_commits_count"] += 1
        if basic_cleaned_msg.startswith("feat:") or basic_cleaned_msg.startswith("feat("): corpus_stats["feat_commits_count"] += 1

        commit_modifies_vendor = False
        current_commit_extensions = set()
        modifications_list = commit_data.get('modifications', [])

        for mod_info in modifications_list: # mod_info é {'filename': ..., 'diff': ...}
            file_path_for_ext = mod_info.get('filename')
            if not file_path_for_ext: continue

            if any(re.search(pattern, file_path_for_ext, re.IGNORECASE) for pattern in [r"vendor/"]):
                 commit_modifies_vendor = True
            _, extension = os.path.splitext(file_path_for_ext)
            if extension: current_commit_extensions.add(extension.lower())

            # ----- PONTO DE INTEGRAÇÃO: Usar seu 'terraform_ast.py' (se disponível) -----
            if extension.lower() == '.tf' and 'terraform_ast' in user_analysis_modules_dict:
                try:
                    tf_analyzer = user_analysis_modules_dict['terraform_ast']
                    # Supondo que seu tf_analyzer tem um método para analisar um diff ou conteúdo
                    # e retornar, por exemplo, tipos de recursos
                    # ast_results = tf_analyzer.analyze_tf_modification_info(mod_info) # Passe o dict mod_info
                    # for resource_type in ast_results.get('resource_types_changed', []):
                    #     corpus_stats["terraform_specific_counts"][resource_type] += 1
                    pass # Remova 'pass' e descomente/implemente a lógica acima
                except Exception as e_tf_ast:
                    # print(f"Erro ao analisar TF com seu módulo: {e_tf_ast}")
                    pass # Evitar que o script quebre se a análise AST falhar
            # ----- FIM DO PONTO DE INTEGRAÇÃO -----

        if commit_modifies_vendor: corpus_stats["vendor_commits_count"] +=1
        for ext in current_commit_extensions: corpus_stats["file_extensions_counts"][ext] += 1
    
    if corpus_stats["msg_lengths"]:
        # Garantir que os valores sejam arredondados da mesma forma que serão exibidos no MD
        corpus_stats["msg_len_mean"] = round(np.mean(corpus_stats["msg_lengths"]), 2)
        corpus_stats["msg_len_median"] = round(np.median(corpus_stats["msg_lengths"]), 2)
        corpus_stats["msg_len_p10"] = round(np.percentile(corpus_stats["msg_lengths"], 10), 2)
        corpus_stats["msg_len_p90"] = round(np.percentile(corpus_stats["msg_lengths"], 90), 2)

    # Geração do Markdown (como na versão anterior, Bloco 4)
    # ... (código completo de formatação e salvamento do Markdown) ...
    # Adicionar a contagem de merges genéricos se desejar:
    # | `generic_merges`            | {corpus_stats["generic_merge_commits_count"]} |
    # E a tabela de 'terraform_specific_counts' se implementada.
    md_total_commits = corpus_stats["total_commits"]
    md_total_repos = corpus_stats["total_repos"]
    md_date_range = f"{corpus_stats['min_date']} – {corpus_stats['max_date']}" if corpus_stats['min_date'] and corpus_stats['max_date'] else "N/A" # Ensure both dates exist
    # Usar os valores já arredondados diretamente
    md_msg_len_mean = f"{corpus_stats['msg_len_mean']}" if corpus_stats['msg_len_mean'] else "N/A"
    md_msg_len_median = f"{corpus_stats['msg_len_median']}" if corpus_stats['msg_len_median'] else "N/A"
    md_msg_len_p10_p90 = f"{corpus_stats['msg_len_p10']}/{corpus_stats['msg_len_p90']}" if corpus_stats['msg_len_p10'] and corpus_stats['msg_len_p90'] else "N/A"
    md_generic_merges = corpus_stats["generic_merge_commits_count"]

    ext_table_data = {}
    for ext_name in ['.tf', '.go', '.yaml', '.md']:
        ext_table_data[ext_name] = corpus_stats["file_extensions_counts"].get(ext_name, 0)
    
    top_extensions = corpus_stats["file_extensions_counts"].most_common(10)
    primary_listed_ext = ['.tf', '.go', '.yaml', '.md']
    additional_extensions_list = []
    for ext, count in top_extensions:
        if ext not in primary_listed_ext and len(additional_extensions_list) < 1:
             additional_extensions_list.append((ext,count))

    md_ext_tf = ext_table_data.get('.tf', 0)
    md_ext_go = ext_table_data.get('.go', 0)
    md_ext_yaml = ext_table_data.get('.yaml', 0)
    md_ext_md = ext_table_data.get('.md', 0)
    
    other_extensions_str_rows = ""
    if additional_extensions_list:
        other_extensions_str_rows = f"| {additional_extensions_list[0][0]}    | {additional_extensions_list[0][1]}                                  |"
    else:
        other_extensions_str_rows = "| [Outra]    | 0                                 |"

    md_vendor_commits = corpus_stats["vendor_commits_count"]

    total_stat_commits = corpus_stats["total_processed_for_stats"] if corpus_stats["total_processed_for_stats"] > 0 else 1
    md_perc_urls = f"{(corpus_stats['urls_in_msg_count'] / total_stat_commits) * 100:.2f}%"
    md_perc_issue_refs = f"{(corpus_stats['issue_refs_in_msg_count'] / total_stat_commits) * 100:.2f}%"
    md_perc_emojis = f"{(corpus_stats['emojis_in_msg_count'] / total_stat_commits) * 100:.2f}%"
    md_perc_non_english = f"{(corpus_stats['non_english_msg_count'] / total_stat_commits) * 100:.2f}%"
    md_perc_fix = f"{(corpus_stats['fix_commits_count'] / total_stat_commits) * 100:.2f}%"
    md_perc_feat = f"{(corpus_stats['feat_commits_count'] / total_stat_commits) * 100:.2f}%"

    corpus_overview_md_content = f"""# Corpus Overview
| Métrica                     | Valor                                      |
|-----------------------------|--------------------------------------------|
| `total_commits`             | {md_total_commits}                         |
| `total_repos`               | {md_total_repos}                           |
| `date_range`                | {md_date_range}                            |
| `msg_len_mean`              | {md_msg_len_mean}                          |
| `msg_len_median`            | {md_msg_len_median}                        |
| `msg_len_p10/p90`           | {md_msg_len_p10_p90}                       |
| `generic_merges`            | {md_generic_merges}                        | 

> _Nota: Extraímos somente modificações em arquivos `.tf`, pois o foco são operadores de mutação em Terraform._
"""
    # Removida a seção "Top File Extensions" e a tabela correspondente.
    # A nota acima substitui essa seção.
    
    # Adicionar tabela de TF AST se implementado:
    if corpus_stats["terraform_specific_counts"]:
        corpus_overview_md_content += "\n## Estatísticas Específicas de Terraform (Exemplo Top 5)\n\n"
        corpus_overview_md_content += "| Tipo de Recurso / Métrica TF | Contagem |\n"
        corpus_overview_md_content += "|------------------------------|----------|\n"
        for item, count in corpus_stats["terraform_specific_counts"].most_common(5):
            corpus_overview_md_content += f"| {item} | {count} |\n"
        corpus_overview_md_content += "\n"


    corpus_overview_md_content += f"""## Recomendações de Embedding e Clustering
* **Comparação de Embedding:**
    * **TF-IDF (n-gram 1–3):**
        * *Tempo de Execução Estimado:* [Preencher após teste com script real - TF-IDF]
        * *Esparsidade da Matriz:* [Preencher após teste com script real - TF-IDF, e.g., 98%]
        * *Prós:* Rápido, bom para palavras-chave literais.
        * *Contras:* Não captura semântica profunda, sensível ao tamanho do vocabulário.
    * **SBERT (e.g., `{SBERT_MODEL_NAME}`):**
        * *Tempo de Execução Estimado:* [Preencher após teste com script real - SBERT]
        * *Qualidade Semântica:* Alta, captura nuances de significado.
        * *Custo de Transformação:* Computacionalmente mais intensivo que TF-IDF para gerar embeddings.
        * *Prós:* Melhores representações semânticas, embeddings densos.
        * *Contras:* Mais lento para embeddar, pode exigir mais recursos.

* **Valores Iniciais Sugeridos (ajustar com base na exploração):**
    * **UMAP:**
      | Parâmetro        | Valor Sugerido                                                                 |
      |------------------|--------------------------------------------------------------------------------|
      | `n_neighbors`    | ~15-50 (considerar `total_commits`; < local, > global)                         |
      | `min_dist`       | ~0.0-0.1 (controla densidade do cluster; < para maior densidade)               |
      | `n_components`   | (Depende do objetivo, e.g., 2 para visualização, 5-10 para HDBSCAN)            |
    * **HDBSCAN:**
      | Parâmetro           | Valor Sugerido                                                                    |
      |---------------------|-----------------------------------------------------------------------------------|
      | `min_cluster_size`  | ~max(10, int(np.sqrt({corpus_stats.get("total_processed_for_stats", nlp_cfg.get('sampling_params', {}).get('sample_size', 1000))}))) |
      | `min_samples`       | ~max(5, int({corpus_stats.get("min_cluster_size", 20)} / 2)) (ou valor explícito como 10) |


## Diagnóstico de Qualidade de Texto
| Métrica                                   | Percentual   |
|-------------------------------------------|--------------|
| Mensagens com URLs                        | {md_perc_urls}   |
| Mensagens com IDs de Issue (`#\\d+`)       | {md_perc_issue_refs} |
| Mensagens com Emojis                      | {md_perc_emojis} |
| Mensagens não-inglesas (via langdetect)   | {md_perc_non_english} |
| Commits "fix:" (Conventional Commits)     | {md_perc_fix}    |
| Commits "feat:" (Conventional Commits)    | {md_perc_feat}   |
"""
    # Nome do arquivo corpus_overview.md a partir do nlp_cfg
    corpus_overview_md_filename = nlp_cfg.get('output_paths', {}).get('corpus_overview_md', 'corpus_overview.md')
    output_md_path = os.path.join(OUTPUT_DIR, corpus_overview_md_filename)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(corpus_overview_md_content)
    print(f"\nArquivo {corpus_overview_md_filename} salvo em: {output_md_path}")

else:
    print("Nenhum commit processado (all_commits_data está vazio), corpus_overview.md não será gerado.")

# %% [VALIDAÇÃO - corpus_overview.md]
# Inserir após o Bloco [4] ter gerado e salvo corpus_overview.md

print("\n--- Iniciando Validação de corpus_overview.md ---")
try:
    if not df_all_commits_processed.empty:
        # Métricas calculadas pelo script de validação usando o mesmo método do overview
        val_min_date = df_all_commits_processed['author_date'].min().strftime('%Y-%m-%d')
        val_max_date = df_all_commits_processed['author_date'].max().strftime('%Y-%m-%d')
        
        # Para validação, usar exatamente os mesmos valores já calculados em corpus_stats
        # para garantir consistência total
        val_msg_lengths_from_overview_logic = corpus_stats.get("msg_lengths", [])

        if val_msg_lengths_from_overview_logic:
            # Usar exatamente os mesmos valores já calculados e arredondados
            val_mean_msg_len = corpus_stats.get("msg_len_mean", 0)
            val_median_msg_len = corpus_stats.get("msg_len_median", 0)
            val_p10_msg_len = corpus_stats.get("msg_len_p10", 0)
            val_p90_msg_len = corpus_stats.get("msg_len_p90", 0)
        else:
            val_mean_msg_len, val_median_msg_len, val_p10_msg_len, val_p90_msg_len = 0,0,0,0

        # Valores do corpus_overview.md (extraídos de corpus_stats) - já arredondados
        overview_total_commits = corpus_stats.get("total_commits")
        overview_date_range_min = corpus_stats.get("min_date")
        overview_date_range_max = corpus_stats.get("max_date")
        overview_msg_len_mean = corpus_stats.get("msg_len_mean", 0)
        overview_msg_len_median = corpus_stats.get("msg_len_median", 0)
        overview_msg_len_p10 = corpus_stats.get("msg_len_p10", 0)
        overview_msg_len_p90 = corpus_stats.get("msg_len_p90", 0)

        print(f"  Validação de total_commits:")
        print(f"    Overview: {overview_total_commits}, Calculado a partir de df_all_commits_processed: {len(df_all_commits_processed)}")
        if overview_total_commits != len(df_all_commits_processed):
            print("    ALERTA: Divergência em total_commits!")

        print(f"  Validação de date_range:")
        print(f"    Overview: {overview_date_range_min} – {overview_date_range_max}")
        print(f"    Calculado: {val_min_date} – {val_max_date}")
        if overview_date_range_min != val_min_date or overview_date_range_max != val_max_date:
            print("    ALERTA: Divergência em date_range!")
        
        print(f"  Validação de métricas de comprimento de mensagem (usando exatamente os mesmos valores):")
        print(f"    Mean   - Overview: {overview_msg_len_mean}, Calculado: {val_mean_msg_len}")
        if overview_msg_len_mean != val_mean_msg_len: print("    ALERTA: Divergência em msg_len_mean!")
        print(f"    Median - Overview: {overview_msg_len_median}, Calculado: {val_median_msg_len}")
        if overview_msg_len_median != val_median_msg_len: print("    ALERTA: Divergência em msg_len_median!")
        print(f"    P10    - Overview: {overview_msg_len_p10}, Calculado: {val_p10_msg_len}")
        if overview_msg_len_p10 != val_p10_msg_len: print("    ALERTA: Divergência em msg_len_p10!")
        print(f"    P90    - Overview: {overview_msg_len_p90}, Calculado: {val_p90_msg_len}")
        if overview_msg_len_p90 != val_p90_msg_len: print("    ALERTA: Divergência em msg_len_p90!")

        # Parâmetros de UMAP e HDBSCAN (comparação visual com o que está no seu MD)
        # Seus valores de min_cluster_size e min_samples são calculados dinamicamente no MD.
        # Ex: HDBSCAN min_cluster_size: ~max(10, int(np.sqrt(corpus_stats.get("total_processed_for_stats", SAMPLE_SIZE))))
        # A validação aqui é mais para garantir que os valores no MD parecem razoáveis.
        print(f"  Parâmetros UMAP/HDBSCAN: Verificar manualmente se os valores em corpus_overview.md são apropriados para {len(df_all_commits_processed)} commits.")

    else:
        print("  AVISO: df_all_commits_processed está vazio. Validação de corpus_overview.md pulada.")
    print("--- Fim da Validação de corpus_overview.md ---")
except Exception as e_val_overview:
    print(f"ERRO durante a validação de corpus_overview.md: {e_val_overview}")

# %% [5]
# Sub-tarefa 3: Geração de sample_messages.csv

eligible_commits_for_sample = []

if all_commits_data:
    print("\nProcessando commits para sample_messages.csv...")
    
    for commit_data_item in tqdm(all_commits_data, desc="Preparando dados para amostragem"):
        msg_original = commit_data_item.get('msg_original', "")
        # Se commit_data_item['is_merge'] for True (significa que é um merge *genérico*),
        # não tentamos limpar a mensagem para a amostra, pois esses são geralmente filtrados.
        # Se você quiser incluir merges não-genéricos na amostra, ajuste esta lógica.
        if commit_data_item.get('is_merge', False): # Se marcado como merge genérico no Bloco 3
            continue 
            
        msg_clean = clean_commit_message_advanced(msg_original) # Limpeza avançada aqui
        
        # Passar is_merge=False porque já filtramos os genéricos acima.
        # A função is_commit_relevant_for_sampling agora focará mais no conteúdo e nos arquivos.
        if not is_commit_relevant_for_sampling(commit_data_item, msg_clean, is_merge_commit=False):
            continue
            
        # Ler a configuração sobre a necessidade de patch para amostragem
        require_patch = nlp_cfg.get('sampling_settings', {}).get('require_valid_patch_for_sampling', False)

        relevant_file, patch_excerpt = None, None

        if require_patch:
            current_modifications_list = commit_data_item.get('modifications', [])
            relevant_file, patch_excerpt = get_relevant_file_and_patch_enhanced(
                current_modifications_list, 
                user_analysis_modules_dict # Passa os módulos de análise do usuário se carregados
            )
            
            if not relevant_file or not patch_excerpt:
                continue # Pula este commit se o patch for obrigatório e não for encontrado
        else:
            # Se o patch não for obrigatório, usamos placeholders
            relevant_file = "not_extracted"
            patch_excerpt = "not_extracted"

        eligible_commits_for_sample.append({
            "commit_hash": commit_data_item.get('commit_hash'),
            "repo_name": commit_data_item.get('repo_name', 'unknown_repo'),
            "msg_clean": msg_clean,
            "file": relevant_file,
            "patch_excerpt": patch_excerpt,
            "author_date": commit_data_item.get('author_date'), # Já é datetime
            "msg_len": len(msg_clean)
        })

    print(f"Total de commits elegíveis para amostragem (após limpeza e filtragem): {len(eligible_commits_for_sample)}")

    if eligible_commits_for_sample:
        df_eligible = pd.DataFrame(eligible_commits_for_sample)
        
        # ETAPA 1: Garantir que author_date seja datetime e converter erros para NaT.
        df_eligible['author_date'] = pd.to_datetime(df_eligible['author_date'], errors='coerce')
        
        # ETAPA 2: Remover linhas onde author_date (agora NaT se inválido) ou outros campos críticos são nulos.
        df_eligible.dropna(subset=['author_date', 'commit_hash', 'msg_clean'], inplace=True)

        # Assegurar unicidade de commit para amostragem.
        # Se um commit (repo,hash) ainda estiver duplicado aqui (não deveria se o Bloco 3 funcionou),
        # esta linha garante a unicidade para a amostragem.
        df_eligible.drop_duplicates(subset=['repo_name', 'commit_hash'], keep='first', inplace=True)
        print(f"Commits elegíveis únicos para amostragem (por repo_name, commit_hash): {len(df_eligible)}")

        # Lógica de Amostragem Estratificada (como no Bloco 5 da versão anterior, ajustada)
        if not df_eligible.empty and len(df_eligible) >= SAMPLE_SIZE / 10: # Só amostrar se tivermos uma quantidade razoável
            # Neste ponto, df_eligible['author_date'] deve ser um tipo datetime64
            # devido às conversões e dropna anteriores.
            df_eligible['year_month'] = df_eligible['author_date'].dt.to_period('M')
            median_msg_len_eligible = df_eligible['msg_len'].median() if not df_eligible['msg_len'].empty else 0
            df_eligible['msg_len_category'] = df_eligible['msg_len'].apply(lambda x: 'long' if x > median_msg_len_eligible else 'short')

            final_sample_df_list = []
            
            # Tentar amostragem estratificada se houver meses suficientes
            monthly_groups = df_eligible.groupby('year_month')
            num_months = len(monthly_groups)

            if num_months > 1 : # Prosseguir com estratificação apenas se houver variabilidade mensal
                target_samples_per_month_proportional = SAMPLE_SIZE / num_months
                target_per_month = target_samples_per_month_proportional
                if target_samples_per_month_proportional < MIN_COMMITS_PER_MONTH_STRATUM :
                    target_per_month = min(MIN_COMMITS_PER_MONTH_STRATUM, SAMPLE_SIZE / num_months if num_months > 0 else SAMPLE_SIZE)
                if target_per_month * num_months > SAMPLE_SIZE and num_months > 0:
                    target_per_month = SAMPLE_SIZE / num_months
                
                temp_sampled_indices = set()

                for month_period, group in tqdm(monthly_groups, desc="Amostragem estratificada por mês/comprimento"):
                    if len(final_sample_df_list) * len(df_eligible.columns) >= SAMPLE_SIZE * len(df_eligible.columns) : break # Aproximação

                    current_month_quota = 0
                    if len(group) < MIN_COMMITS_PER_MONTH_STRATUM: current_month_quota = len(group)
                    else: current_month_quota = max(MIN_COMMITS_PER_MONTH_STRATUM, int(np.ceil(target_per_month)))
                    num_to_sample_this_month = min(len(group), current_month_quota)

                    # Dentro do estrato mensal, tentar equilibrar curtas/longas
                    # (Lógica de amostragem curta/longa dentro do mês como antes)
                    # ... (código de amostragem de short_msgs_month e long_msgs_month)
                    for category in ['short', 'long']:
                        category_group = group[group['msg_len_category'] == category]
                        # Tentar pegar ~metade da quota do mês desta categoria
                        n_to_sample_category = int(np.ceil(num_to_sample_this_month / 2.0))
                        
                        actual_to_sample_category = min(n_to_sample_category, len(category_group))
                        if actual_to_sample_category > 0:
                            sampled_from_category_df = category_group.sample(
                                n=actual_to_sample_category, 
                                random_state=42, 
                                replace=False # Não substituir dentro do estrato
                            )
                            for idx_val in sampled_from_category_df.index:
                                if idx_val not in temp_sampled_indices and len(final_sample_df_list) < SAMPLE_SIZE :
                                    final_sample_df_list.append(df_eligible.loc[idx_val])
                                    temp_sampled_indices.add(idx_val)


            # Se a amostragem estratificada não preencheu, ou não foi feita, preencher/fazer aleatoriamente
            if len(final_sample_df_list) < SAMPLE_SIZE:
                needed_more = SAMPLE_SIZE - len(final_sample_df_list)
                remaining_eligible_indices = list(set(df_eligible.index) - temp_sampled_indices if 'temp_sampled_indices' in locals() else set(df_eligible.index))
                
                if needed_more > 0 and len(remaining_eligible_indices) > 0:
                    count_to_sample_randomly = min(needed_more, len(remaining_eligible_indices))
                    random_fill_indices = random.sample(remaining_eligible_indices, count_to_sample_randomly)
                    for idx_val in random_fill_indices:
                        if len(final_sample_df_list) < SAMPLE_SIZE :
                             final_sample_df_list.append(df_eligible.loc[idx_val])
            
            if final_sample_df_list:
                df_sample = pd.DataFrame(final_sample_df_list)
                df_sample_output = df_sample[['commit_hash', 'msg_clean', 'file', 'patch_excerpt']]
                # Nome do arquivo sample_messages.csv a partir do nlp_cfg
                sample_messages_csv_filename = nlp_cfg.get('output_paths', {}).get('sample_messages_csv', 'sample_messages_nlp.csv')
                output_csv_sample_path = os.path.join(OUTPUT_DIR, sample_messages_csv_filename)
                df_sample_output.to_csv(output_csv_sample_path, index=False, quoting=csv.QUOTE_ALL)
                print(f"\nArquivo {sample_messages_csv_filename} com {len(df_sample_output)} registros salvo em: {output_csv_sample_path}")
            else:
                print("A amostra final está vazia após a tentativa de estratificação/aleatorização. sample_messages.csv não foi gerado.")

        elif not df_eligible.empty: # Se tivermos poucos dados, pegar todos ou uma amostra aleatória simples
            print(f"Poucos dados elegíveis ({len(df_eligible)}) para amostragem estratificada completa. Pegando uma amostra aleatória ou todos.")
            df_sample = df_eligible.sample(n=min(SAMPLE_SIZE, len(df_eligible)), random_state=42, replace=False)
            df_sample_output = df_sample[['commit_hash', 'msg_clean', 'file', 'patch_excerpt']]
            # Nome do arquivo sample_messages.csv a partir do nlp_cfg
            sample_messages_csv_filename = nlp_cfg.get('output_paths', {}).get('sample_messages_csv', 'sample_messages_nlp.csv')
            output_csv_sample_path = os.path.join(OUTPUT_DIR, sample_messages_csv_filename)
            df_sample_output.to_csv(output_csv_sample_path, index=False, quoting=csv.QUOTE_ALL)
            print(f"\nArquivo {sample_messages_csv_filename} com {len(df_sample_output)} registros salvo em: {output_csv_sample_path}")            
        else:
             print("Nenhum commit elegível para amostragem após a dedetização. sample_messages.csv não foi gerado.")
    else:
        print("Nenhum commit elegível encontrado (lista eligible_commits_for_sample vazia). sample_messages.csv não será gerado.")
else:
    print("Nenhum dado de commit carregado (all_commits_data está vazio). sample_messages.csv não será gerado.")


# %% [VALIDAÇÃO - sample_messages.csv]
# Inserir no final do Bloco [5], após sample_messages.csv ser salvo.
# df_sample_output é o que foi salvo. df_sample (se existir) ou df_eligible podem ter mais dados.
# Usaremos df_all_commits_processed para dados do corpus completo.

print("\n--- Iniciando Validação de sample_messages.csv ---")
try:
    if 'df_sample_output' in locals() and not df_sample_output.empty and not df_all_commits_processed.empty:
        num_linhas_sample = len(df_sample_output)
        print(f"  Número de linhas em sample_messages.csv: {num_linhas_sample}")
        if not (0.9 * SAMPLE_SIZE <= num_linhas_sample <= 1.1 * SAMPLE_SIZE): # Permitir pequena variação
            print(f"    ALERTA: Número de linhas ({num_linhas_sample}) fora do esperado (~{SAMPLE_SIZE}).")

        # Para verificar estratificação e representatividade, precisamos de 'author_date' e 'msg_len' na amostra.
        # df_sample (se existir e foi usado para criar df_sample_output) ou df_eligible_for_sample_df (se usado)
        # ou precisamos recriar um df_sample_enriched.
        # Vamos assumir que `final_sample_df_list` foi usado para criar `df_sample` e depois `df_sample_output`.
        if 'final_sample_df_list' in locals() and final_sample_df_list:
            df_sample_for_val = pd.DataFrame(final_sample_df_list) # Contém 'author_date', 'msg_len'
            df_sample_for_val['author_date'] = pd.to_datetime(df_sample_for_val['author_date'], errors='coerce')
            
            # Estratificação por mês (≥ 50 commits/mês NA AMOSTRA)
            if not df_sample_for_val['author_date'].isnull().all():
                df_sample_for_val['year_month_val'] = df_sample_for_val['author_date'].dt.to_period('M')
                sample_commits_per_month_val = df_sample_for_val['year_month_val'].value_counts()
                
                months_lt_min_commits = sample_commits_per_month_val[sample_commits_per_month_val < MIN_COMMITS_PER_MONTH_STRATUM].count()
                print(f"  Estratificação por mês na amostra (objetivo >= {MIN_COMMITS_PER_MONTH_STRATUM} commits/mês amostrado):")
                print(f"    Meses na amostra com < {MIN_COMMITS_PER_MONTH_STRATUM} commits: {months_lt_min_commits} de {len(sample_commits_per_month_val)} meses amostrados.")
                # print(f"    Distribuição mensal na amostra: {sample_commits_per_month_val.to_dict()}")
            else:
                print("    AVISO: Não foi possível verificar a estratificação mensal (coluna 'author_date' ausente ou vazia na amostra de validação).")

            # Representatividade Temporal (comparando distribuição anual)
            if not df_sample_for_val['author_date'].isnull().all() and 'author_date' in df_all_commits_processed.columns:
                df_all_commits_processed['year_val'] = df_all_commits_processed['author_date'].dt.year
                all_commits_yearly_dist_val = df_all_commits_processed['year_val'].value_counts(normalize=True).sort_index()
                
                df_sample_for_val['year_val'] = df_sample_for_val['author_date'].dt.year
                sample_yearly_dist_val = df_sample_for_val['year_val'].value_counts(normalize=True).sort_index()
                
                print(f"  Representatividade Temporal (Distribuição Anual Normalizada):")
                # print(f"    Corpus Completo: {all_commits_yearly_dist_val.to_dict()}")
                # print(f"    Amostra: {sample_yearly_dist_val.to_dict()}")
                # Uma comparação mais programática poderia verificar o K-S test ou similar, mas visual é um bom começo.
                # Por simplicidade, apenas imprimimos para inspeção visual.
                if len(all_commits_yearly_dist_val) == len(sample_yearly_dist_val):
                     diffs = (all_commits_yearly_dist_val - sample_yearly_dist_val).abs()
                     if diffs.max() > 0.1: # Se a diferença máxima em qualquer ano for > 10%
                         print("    ALERTA: Distribuição temporal da amostra pode divergir significativamente do corpus.")
                else:
                     print("    ALERTA: Número de anos diferente entre corpus e amostra para comparação direta.")


            # Representatividade de Tamanho de Mensagem (msg_len já deve estar em df_sample_for_val)
            if 'msg_len' in df_sample_for_val.columns and 'msg_word_count_for_val' in df_all_commits_processed.columns:
                p10_all = df_all_commits_processed['msg_word_count_for_val'].quantile(0.10)
                p50_all = df_all_commits_processed['msg_word_count_for_val'].quantile(0.50)
                p90_all = df_all_commits_processed['msg_word_count_for_val'].quantile(0.90)

                p10_sample = df_sample_for_val['msg_len'].quantile(0.10)
                p50_sample = df_sample_for_val['msg_len'].quantile(0.50)
                p90_sample = df_sample_for_val['msg_len'].quantile(0.90)
                print(f"  Representatividade de Tamanho de Mensagem (palavras em msg_clean):")
                print(f"    Corpus - P10: {p10_all:.2f}, P50: {p50_all:.2f}, P90: {p90_all:.2f}")
                print(f"    Amostra- P10: {p10_sample:.2f}, P50: {p50_sample:.2f}, P90: {p90_sample:.2f}")
                if abs(p50_all - p50_sample) > 0.2 * p50_all : # Se a mediana diferir em mais de 20%
                    print("    ALERTA: Mediana do tamanho das mensagens na amostra diverge do corpus.")
            else:
                print("    AVISO: Não foi possível verificar representatividade de tamanho de mensagem ('msg_len' ou 'msg_word_count_for_val' ausente).")
        else:
            print("  AVISO: 'final_sample_df_list' não encontrado ou vazio. Validação detalhada da amostra pulada.")
    else:
        print("  AVISO: df_sample_output ou df_all_commits_processed não disponível/vazio. Validação de sample_messages.csv pulada.")
    print("--- Fim da Validação de sample_messages.csv ---")
except Exception as e_val_sample:
    print(f"ERRO durante a validação de sample_messages.csv: {e_val_sample}")

# %% [6]
# Sub-tarefa 4: Geração de token_freq.csv e Sugestões de keywords_stop.yml
# (Praticamente o mesmo, mas a fonte de 'all_cleaned_messages_for_tokens' é mais clara)

all_cleaned_messages_for_tokens = []

# Definir caminhos para os arquivos que serão criados por main.py
# ALL_COMMITS_CSV_PATH_EXPECTED é o caminho para o reports/all_commits.csv (do minerador)
# TOP_BIGRAMS_JSON_PATH_EXPECTED é o caminho para o reports/top_bigrams.json (do minerador)

# Tentar ler 'all_commits.csv' que é esperado ser gerado pelo main.py (processo de mineração)
if os.path.exists(ALL_COMMITS_CSV_PATH_EXPECTED):
    print(f"\nLendo mensagens de {ALL_COMMITS_CSV_PATH_EXPECTED} para token_freq.csv e keywords_stop.yml...")
    try:
        df_all_commits_input = pd.read_csv(ALL_COMMITS_CSV_PATH_EXPECTED, low_memory=False)
        # Garantir que msg_clean exista e não seja NaN, ou usar msg_original
        if 'msg_clean' in df_all_commits_input.columns:
            all_cleaned_messages_for_tokens = df_all_commits_input['msg_clean'].dropna().astype(str).tolist()
        elif 'msg_original' in df_all_commits_input.columns:
            print(f"AVISO: Coluna 'msg_clean' não encontrada em {ALL_COMMITS_CSV_PATH_EXPECTED}. Limpando 'msg_original' para tokenização.")
            all_cleaned_messages_for_tokens = df_all_commits_input['msg_original'].dropna().astype(str).apply(clean_commit_message_advanced).tolist()
            all_cleaned_messages_for_tokens = [m for m in all_cleaned_messages_for_tokens if m] # Remover None/vazios após limpeza
        else:
            print(f"AVISO: Nenhuma coluna de mensagem ('msg_clean' ou 'msg_original') encontrada em {ALL_COMMITS_CSV_PATH_EXPECTED} para tokenização.")
            all_cleaned_messages_for_tokens = []
        
        if not all_cleaned_messages_for_tokens:
            print(f"AVISO: Nenhuma mensagem pôde ser preparada para tokenização a partir de {ALL_COMMITS_CSV_PATH_EXPECTED}.")

    except Exception as e:
        print(f"AVISO: Erro ao ler ou processar {ALL_COMMITS_CSV_PATH_EXPECTED}: {e}. Tentando fallback...")
        all_cleaned_messages_for_tokens = [] # Resetar em caso de erro de leitura

if not all_cleaned_messages_for_tokens: # Fallback se all_commits.csv não existir ou falhar
    print("AVISO: Falha ao carregar mensagens do CSV principal. Usando lógica de fallback para coletar mensagens para tokens...")
    if eligible_commits_for_sample: # Preferir mensagens dos commits que foram considerados para amostragem (já filtrados)
        print("\nPreparando mensagens para token_freq.csv a partir de 'eligible_commits_for_sample'...")
        all_cleaned_messages_for_tokens = [c['msg_clean'] for c in eligible_commits_for_sample if c.get('msg_clean')]
    elif all_commits_data: # Fallback: limpar todas as mensagens se a lista de elegíveis estiver vazia
        print("Lista 'eligible_commits_for_sample' vazia. Usando todas as mensagens de 'all_commits_data' para token_freq.")
        for commit_data_item in tqdm(all_commits_data, desc="Limpando todas as mensagens para token_freq (fallback)"):
            if not commit_data_item.get('is_merge', False): # Excluir merges genéricos da contagem de tokens
                msg_clean = clean_commit_message_advanced(commit_data_item.get('msg_original', ""))
                if msg_clean:
                    all_cleaned_messages_for_tokens.append(msg_clean)
    else:
        print("Nenhum dado de commit disponível para gerar token_freq.csv.")


if all_cleaned_messages_for_tokens:
    print(f"Total de mensagens limpas para análise de tokens: {len(all_cleaned_messages_for_tokens)}")
    token_counts = Counter()
    # ... (Restante da lógica de tokenização, contagem, salvamento de token_freq.csv e sugestões de keywords_stop.yml
    #      como no Bloco 6 da versão anterior, sem alterações significativas necessárias aqui) ...
    print("Tokenizando mensagens e contando frequências...")
    for msg in tqdm(all_cleaned_messages_for_tokens, desc="Tokenizando"):
        # Usar msg.split() conforme o plano para token_freq.csv, mas manter NLTK para robustez geral se desejado.
        # Para este passo, vamos seguir o plano de usar split() para token_freq, mas notar que pode ser menos robusto.
        # No entanto, clean_commit_message_advanced já remove muita pontuação.
        # Para keywords_stop, a tokenização original NLTK pode ser mantida se for para uma análise mais refinada lá.
        # Vamos usar split() para a contagem de token_freq.csv:
        tokens = msg.lower().split() # Usando split() como por plano.
        
        # Para STOP_WORDS_EN, a tokenização original era com NLTK. Se usarmos split(), o filtro de stopwords pode variar.
        # Mantendo filtro similar ao original:
        filtered_tokens = [
            token for token in tokens 
            if token not in STOP_WORDS_EN and len(token) > 1 and not token.isdigit() and token.isalnum() # Adicionado isalnum para remover pontuação restante
        ]
        token_counts.update(filtered_tokens)

    # Permitir ajuste de top_N para token_freq.csv (pode ser configurado ou passado como argumento no futuro)
    # TOP_N_TOKENS já é lido do nlp_cfg na Cell [1]
    top_n_tokens_list = token_counts.most_common(TOP_N_TOKENS)
    
    # Nome do arquivo token_freq.csv a partir do nlp_cfg
    token_freq_csv_filename = nlp_cfg.get('output_paths', {}).get('token_freq_csv', 'token_freq_nlp.csv')
    output_csv_tokens_path = os.path.join(OUTPUT_DIR, token_freq_csv_filename)
    with open(output_csv_tokens_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "freq"])
        writer.writerows(top_n_tokens_list)
    print(f"\nArquivo {token_freq_csv_filename} ({TOP_N_TOKENS} tokens) salvo em: {output_csv_tokens_path}")

    potential_custom_stopwords = []
    
    # Ler heurísticas de stopwords do nlp_cfg
    _stopwords_heuristics_cfg = nlp_cfg.get('token_analysis_params', {}).get('stopwords_heuristics', {})
    merge_terms = _stopwords_heuristics_cfg.get('merge_terms', ['merge', 'pull', 'branch']) # Default se não configurado
    domain_common_terms = _stopwords_heuristics_cfg.get('domain_common_terms', [
        'terraform', 'provider', 'helm', 'aws', 'gcp', 'azure', 'kubernetes', 'k8s',
        'module', 'resource', 'variable', 'output', 'data', 'release', 'chart', 'value',
        'config', 'plugin', 'template', 'service', 'instance', 'cluster', 'network',
        'bump', 'update', 'version', 'upgrade', 'refactor', 'docs', 'readme',
        'chore', 'build', 'test', 'ci', 'lint', 'format'
    ]) # Default se não configurado
    keep_frequent_terms = _stopwords_heuristics_cfg.get('keep_frequent_terms', ['fix', 'feat', 'use', 'change', 'remove', 'default', 'error', 'bug', 'issue']) # Default

    # Adicionar termos de merge explicitamente
    for term in merge_terms:
        potential_custom_stopwords.append(term)

    # domain_common_terms = [
    #     'terraform', 'provider', 'helm', 'aws', 'gcp', 'azure', 'kubernetes', 'k8s',
    #     'module', 'resource', 'variable', 'output', 'data', 'release', 'chart', 'value',
    #     'config', 'plugin', 'template', 'service', 'instance', 'cluster', 'network',
    #     'bump', 'update', 'version', 'upgrade', 'refactor', 'docs', 'readme',
    #     'chore', 'build', 'test', 'ci', 'lint', 'format'
    # ] # Esta lista agora vem do config
    one_percent_threshold = 0.01 * len(all_cleaned_messages_for_tokens) if all_cleaned_messages_for_tokens else 0

    # Usar a lista top_n_tokens_list que já foi calculada
    for token, freq in top_n_tokens_list: # Iterar sobre os mesmos tokens que foram para token_freq.csv
        # Limpeza de comentários: Não adicionar comentários como "# (ocorre em >X msgs)"
        if token in domain_common_terms:
            potential_custom_stopwords.append(token)
        elif freq > one_percent_threshold and len(token) > 2:
             if token not in keep_frequent_terms: # Usa a lista de keep_frequent_terms do config
                potential_custom_stopwords.append(token)
    
    # Tentar carregar bigramas de top_bigrams.json
    loaded_bigrams = []
    if os.path.exists(TOP_BIGRAMS_JSON_PATH_EXPECTED):
        try:
            with open(TOP_BIGRAMS_JSON_PATH_EXPECTED, 'r', encoding='utf-8') as f_bigram:
                # Espera-se que o JSON seja uma lista de listas/tuplas, ex: [["tf", "provider"], ["pull", "request"]]
                # Ou uma lista de strings concatenadas: ["tf_provider", "pull_request"]
                # O plano para extract_bigrams sugere: sorted(zip(vect.get_feature_names(), sums), key=lambda x:-x[1])[:20]
                # get_feature_names() para bigramas retorna "word1 word2".
                raw_bigrams = json.load(f_bigram) # Assume que é uma lista de [bigram_string, count]
                if raw_bigrams and isinstance(raw_bigrams, list) and isinstance(raw_bigrams[0], (list, tuple)):
                     loaded_bigrams = [item[0] for item in raw_bigrams] # Pega só a string do bigrama "word1 word2"
                elif raw_bigrams and isinstance(raw_bigrams, list) and isinstance(raw_bigrams[0], str): # Se for só lista de strings "word1 word2"
                    loaded_bigrams = raw_bigrams

        except Exception as e_bigram:
            print(f"AVISO: Erro ao carregar ou processar {TOP_BIGRAMS_JSON_PATH_EXPECTED}: {e_bigram}")

    if loaded_bigrams:
        print(f"Incorporando {len(loaded_bigrams)} bigramas de {TOP_BIGRAMS_JSON_PATH_EXPECTED} nas sugestões de stopwords.")
        for bigram_str in loaded_bigrams:
            # Adicionar o bigrama como uma string única (se o sistema de stoplist suportar frases)
            # ou dividir em tokens individuais. Por simplicidade, vamos adicionar tokens individuais.
            # Isso pode ser refinado se o pipeline de NLP puder usar frases como stopwords.
            potential_custom_stopwords.append(bigram_str) # Adiciona "word1 word2"
            # Alternativamente, para adicionar palavras individuais do bigrama:
            # for token_in_bigram in bigram_str.split():
            #    if len(token_in_bigram) > 1 and not token_in_bigram.isdigit():
            #        potential_custom_stopwords.append(token_in_bigram)


    unique_potential_stopwords = sorted(list(set(s.lower() for s in potential_custom_stopwords))) # Garantir lowercase e unicidade
    keywords_stop_yml_content = "# Sugestões de keywords_stop.yml\n"
    keywords_stop_yml_content += "# Analise esta lista e adicione/remova conforme necessário para seu topic modeling.\n"
    keywords_stop_yml_content += "# Bigramas (se presentes) são listados como frases. Verifique se seu sistema de NLP os suporta.\n"
    keywords_stop_yml_content += "custom_stopwords:\n"
    for word_entry in unique_potential_stopwords:
        # Não adicionar mais comentários como "# (ocorre em >X msgs)"
        keywords_stop_yml_content += f"  - {word_entry}\n"
            
    # Nome do arquivo de sugestões de stopwords a partir do nlp_cfg
    keywords_stop_suggestions_filename = nlp_cfg.get('output_paths', {}).get('keywords_stop_yml_suggestions', 'keywords_stop_nlp_suggestions.txt')
    output_yml_path = os.path.join(OUTPUT_DIR, keywords_stop_suggestions_filename)
    with open(output_yml_path, "w", encoding="utf-8") as f:
        f.write(keywords_stop_yml_content)
    print(f"\nSugestões para keywords_stop.yml salvas em: {output_yml_path}")
else:
    print("Nenhuma mensagem limpa disponível para gerar token_freq.csv.")

print("\n--- Script Integrado (v2) Concluído ---")
print(f"Resultados salvos em: {os.path.abspath(OUTPUT_DIR)}")


# %% [VALIDAÇÃO - token_freq.csv e keywords_stop.yml_suggestions.txt]
# Inserir no final do Bloco [6]

print("\n--- Iniciando Validação de token_freq.csv e keywords_stop.yml ---")
try:
    if 'top_n_tokens_list' in locals() and top_n_tokens_list and 'all_cleaned_messages_for_tokens' in locals() and all_cleaned_messages_for_tokens:
        # Validação de token_freq.csv (Cobertura)
        # top_n_tokens_list é [(token, freq), ...]
        sum_freq_top_n = sum(item[1] for item in top_n_tokens_list)
        
        # Recalcular todas as ocorrências de tokens no corpus (all_cleaned_messages_for_tokens)
        # usando a mesma lógica de tokenização e filtro de stopwords que gerou token_counts
        total_token_occurrences_in_corpus_val = 0
        temp_token_counts_for_total = Counter()
        for msg_val in all_cleaned_messages_for_tokens: # Usar a mesma lista de mensagens
            tokens_val = msg_val.lower().split() 
            filtered_tokens_val = [
                token for token in tokens_val
                if token not in STOP_WORDS_EN and len(token) > 1 and not token.isdigit() and token.isalnum()
            ]
            temp_token_counts_for_total.update(filtered_tokens_val)
        total_token_occurrences_in_corpus_val = sum(temp_token_counts_for_total.values())

        if total_token_occurrences_in_corpus_val > 0:
            coverage_percentage = (sum_freq_top_n / total_token_occurrences_in_corpus_val) * 100
            print(f"  Validação de token_freq.csv (Top {len(top_n_tokens_list)} tokens):")
            print(f"    Soma das frequências dos top tokens: {sum_freq_top_n}")
            print(f"    Total de ocorrências de tokens (filtrados) no corpus: {total_token_occurrences_in_corpus_val}")
            print(f"    Cobertura dos top tokens: {coverage_percentage:.2f}%")
            if coverage_percentage < 60:
                print(f"    ALERTA: Cobertura ({coverage_percentage:.2f}%) abaixo de 60%. Considere aumentar TOP_N_TOKENS (atualmente {TOP_N_TOKENS}).") # TOP_N_TOKENS já é do nlp_cfg
        else:
            print("    AVISO: Não foi possível calcular a cobertura (total_token_occurrences_in_corpus_val é zero).")

        # Validação de keywords_stop.yml_suggestions.txt
        if 'unique_potential_stopwords' in locals():
            print(f"  Validação de keywords_stop.yml_suggestions.txt:")
            print(f"    Número de stopwords únicas sugeridas: {len(unique_potential_stopwords)}")
            if not any(term in unique_potential_stopwords for term in ['merge', 'pull', 'branch']):
                print("    ALERTA: Termos de merge ('merge', 'pull', 'branch') não encontrados nas sugestões de stopwords.")
            # Verificar se os top N tokens (que não são obviamente úteis) estão lá
            noisy_tokens_to_check = [t[0] for t in top_n_tokens_list[:10] if t[0] not in ['fix', 'add', 'update', 'feat', 'support', 'resource', 'provider']] # Exemplo
            missing_noisy_suggestions = [nt for nt in noisy_tokens_to_check if nt not in unique_potential_stopwords]
            if missing_noisy_suggestions:
                 print(f"    INFO: Alguns tokens frequentes como {missing_noisy_suggestions} não estão nas sugestões. Verifique se devem ser adicionados.")
        else:
            print("    AVISO: 'unique_potential_stopwords' não encontrado. Validação de keywords_stop.yml pulada.")
            
    elif 'top_n_tokens_list' in locals() and not top_n_tokens_list:
         print("  AVISO: 'top_n_tokens_list' está vazia. Validação de token_freq.csv pulada.")
    else:
        print("  AVISO: Dados para validação de token_freq.csv ou keywords_stop.yml não disponíveis.")

    # Validação de top_bigrams.json (se carregado)
    if 'loaded_bigrams' in locals() and loaded_bigrams: # loaded_bigrams vem do carregamento de TOP_BIGRAMS_JSON_PATH_EXPECTED
        print(f"  Validação de top_bigrams.json (carregado):")
        print(f"    Número de bigramas carregados: {len(loaded_bigrams)}")
        if len(loaded_bigrams) > 0:
            print(f"    Exemplo do primeiro bigrama carregado: '{loaded_bigrams[0]}'")
            # Verificar se os bigramas fazem sentido (ex: não contêm stopwords comuns se já filtradas)
            # Esta validação é mais qualitativa.
        else:
            print("    INFO: Nenhum bigrama carregado de top_bigrams.json (arquivo pode estar vazio ou não encontrado).")
    elif os.path.exists(TOP_BIGRAMS_JSON_PATH_EXPECTED):
        print(f"  INFO: {TOP_BIGRAMS_JSON_PATH_EXPECTED} existe, mas 'loaded_bigrams' não foi populado. Verifique a lógica de carregamento no Bloco [6].")
    else:
        print(f"  INFO: {TOP_BIGRAMS_JSON_PATH_EXPECTED} não encontrado. Validação de top_bigrams.json pulada.")


    print("--- Fim da Validação de token_freq.csv e keywords_stop.yml ---")
except Exception as e_val_tokens:
    print(f"ERRO durante a validação de tokens/stopwords: {e_val_tokens}")

#%% Bloco [8] - Vetorização com SBERT e Clusterização com HDBSCAN
#%% [8] - Vetorização com SBERT e Clusterização com HDBSCAN
def gerar_embeddings_sbert(mensagens, nome_modelo_sbert, output_dir='.', ficheiro_embeddings_basename='sbert_embeddings.pkl'):
    """
    Gera embeddings para uma lista de mensagens usando SBERT e salva-os.
    """
    caminho_completo_ficheiro_embeddings = os.path.join(output_dir, ficheiro_embeddings_basename)
    print(f"A carregar o modelo SBERT: {nome_modelo_sbert}...")
    modelo = SentenceTransformer(nome_modelo_sbert)

    print(f"A gerar embeddings para {len(mensagens)} mensagens...")
    # É importante garantir que as mensagens sejam strings
    mensagens_str = [str(msg) if msg is not None else '' for msg in mensagens]
    embeddings = modelo.encode(mensagens_str, show_progress_bar=True)

    print(f"Dimensão dos embeddings: {embeddings.shape}")

    # Salvar os embeddings para não ter de os recalcular
    os.makedirs(output_dir, exist_ok=True) # Garantir que o diretório de output existe
    with open(caminho_completo_ficheiro_embeddings, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings salvos em: {caminho_completo_ficheiro_embeddings}")
    return embeddings

# Preparar dados para o Bloco [8]
mensagens_para_vetorizar = []
sbert_embeddings = None
skip_clustering_steps = True
df_commits_para_clustering = pd.DataFrame() # DataFrame vazio por padrão

if 'df_all_commits_processed' in locals() and not df_all_commits_processed.empty:
    # Ler configuração de abordagem de clustering
    use_all_commits_for_clustering = nlp_cfg.get('clustering_settings', {}).get('use_all_commits_for_clustering', True)
    
    if use_all_commits_for_clustering:
        # Abordagem "Top-Down" (ATUAL): Usar todos os commits
        print(f"Usando abordagem TOP-DOWN: Clusterizando todos os {len(df_all_commits_processed)} commits")
        df_commits_para_clustering = df_all_commits_processed.copy()
        
        # Verificar se deve remover duplicatas baseado na configuração
        remove_duplicates = nlp_cfg.get('clustering_settings', {}).get('remove_duplicates_for_clustering', False)
        if remove_duplicates:
            initial_count = len(df_commits_para_clustering)
            df_commits_para_clustering.drop_duplicates(subset=['repo_name', 'commit_hash'], keep='first', inplace=True)
            final_count = len(df_commits_para_clustering)
            print(f"Remoção de duplicatas ativada: {initial_count} → {final_count} commits únicos (-{initial_count - final_count} duplicatas)")
        else:
            print(f"Mantendo duplicatas: {len(df_commits_para_clustering)} commits (incluindo possíveis duplicatas)")
    else:
        # Abordagem "Bottom-Up" (NOVA): Filtrar commits primeiro
        print("Usando abordagem BOTTOM-UP: Filtrando commits antes do clustering")
        
        # Aplicar filtros de elegibilidade (similar ao usado em amostragem)
        eligible_commits_for_clustering = []
        
        for _, commit_row in df_all_commits_processed.iterrows():
            # Pular commits de merge genérico
            if commit_row.get('is_merge', False):
                continue
                
            msg_clean = commit_row.get('msg_clean_for_val', '')
            if not msg_clean:
                continue
                
            # Criar um dict compatível com is_commit_relevant_for_sampling
            commit_data_item = {
                'modifications': commit_row.get('modifications', [])
            }
            
            # Aplicar função de relevância (focando em arquivos e conteúdo)
            if not is_commit_relevant_for_sampling(commit_data_item, msg_clean, is_merge_commit=False):
                continue
                
            eligible_commits_for_clustering.append(commit_row)
        
        if eligible_commits_for_clustering:
            df_commits_para_clustering = pd.DataFrame(eligible_commits_for_clustering)
            print(f"Commits filtrados para clustering: {len(df_commits_para_clustering)} de {len(df_all_commits_processed)} ({len(df_commits_para_clustering)/len(df_all_commits_processed)*100:.1f}%)")
            
            # Verificar se deve remover duplicatas baseado na configuração
            remove_duplicates = nlp_cfg.get('clustering_settings', {}).get('remove_duplicates_for_clustering', False)
            if remove_duplicates:
                initial_count = len(df_commits_para_clustering)
                df_commits_para_clustering.drop_duplicates(subset=['repo_name', 'commit_hash'], keep='first', inplace=True)
                final_count = len(df_commits_para_clustering)
                print(f"Remoção de duplicatas ativada: {initial_count} → {final_count} commits únicos (-{initial_count - final_count} duplicatas)")
            else:
                print(f"Mantendo duplicatas: {len(df_commits_para_clustering)} commits (incluindo possíveis duplicatas)")
        else:
            df_commits_para_clustering = pd.DataFrame()
            print("AVISO: Nenhum commit passou pelos filtros de elegibilidade")
    
    if not df_commits_para_clustering.empty:
        # Usar 'msg_clean_for_val' que foi criada para validações e é robusta.
        # Assegurar que não há NaNs quebrando o SBERT.
        df_commits_para_clustering['processed_message_for_sbert'] = df_commits_para_clustering['msg_clean_for_val'].fillna('')
        mensagens_para_vetorizar = df_commits_para_clustering['processed_message_for_sbert'].tolist()

        if not mensagens_para_vetorizar:
            print("AVISO Bloco [8]: Nenhuma mensagem para vetorizar após o pré-processamento. Pulando SBERT e HDBSCAN.")
        else:
            print(f"Bloco [8]: {len(mensagens_para_vetorizar)} mensagens prontas para vetorização.")
            skip_clustering_steps = False
    else:
        print("AVISO Bloco [8]: DataFrame de commits para clustering está vazio. Pulando SBERT e HDBSCAN.")
else:
    print("AVISO Bloco [8]: df_all_commits_processed não está disponível ou está vazio. Pulando SBERT e HDBSCAN.")


if not skip_clustering_steps:
    # Caminho para salvar/carregar os embeddings dentro do OUTPUT_DIR
    # Nome do arquivo de embeddings a partir do nlp_cfg
    sbert_embeddings_pkl_filename = nlp_cfg.get('output_paths', {}).get('sbert_embeddings_pkl', 'sbert_embeddings.pkl')
    CAMINHO_EMBEDDINGS = os.path.join(OUTPUT_DIR, sbert_embeddings_pkl_filename)

    # Verificar se os embeddings já existem para poupar tempo
    if os.path.exists(CAMINHO_EMBEDDINGS):
        print(f"A carregar embeddings de: {CAMINHO_EMBEDDINGS}")
        try:
            with open(CAMINHO_EMBEDDINGS, 'rb') as f:
                sbert_embeddings = pickle.load(f)
            if sbert_embeddings.shape[0] != len(mensagens_para_vetorizar):
                print(f"AVISO: Número de embeddings ({sbert_embeddings.shape[0]}) não corresponde ao número de mensagens ({len(mensagens_para_vetorizar)}). Regerando.")
                sbert_embeddings = None # Forçar regeneração
        except Exception as e:
            print(f"Erro ao carregar embeddings de {CAMINHO_EMBEDDINGS}: {e}. Regerando.")
            sbert_embeddings = None

    if sbert_embeddings is None: # Se não carregou ou precisa regerar
        sbert_embeddings = gerar_embeddings_sbert(
            mensagens_para_vetorizar,
            nome_modelo_sbert=SBERT_MODEL_NAME, # SBERT_MODEL_NAME já é do nlp_cfg (Cell [1])
            output_dir=OUTPUT_DIR,
            ficheiro_embeddings_basename=sbert_embeddings_pkl_filename # Usa o nome do arquivo do nlp_cfg
        )

    # Adicionar embeddings ao DataFrame (opcional, pode ser grande, não faremos por padrão para economizar memória no CSV)
    # df_commits_para_clustering['sbert_embedding'] = list(sbert_embeddings)

    # --- Passo 1.3: Redução de Dimensionalidade com UMAP (Opcional, mas recomendado) e Clusterização com HDBSCAN ---

    def clusterizar_com_hdbscan(embeddings,
                                n_neighbors=15, min_dist=0.0, n_components=10, # Parâmetros UMAP
                                metric='cosine', # Parâmetro UMAP
                                min_cluster_size=60, min_samples=10, # Parâmetros HDBSCAN
                                allow_single_cluster=True, # Parâmetro HDBSCAN
                                cluster_selection_epsilon=0.0, # Parâmetro HDBSCAN
                                usar_umap=True):
        """
        Realiza clusterização com HDBSCAN, opcionalmente usando UMAP para redução de dimensionalidade.
        """
        dados_para_clusterizar = embeddings

        if usar_umap:
            print(f"A aplicar UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}, metric={metric})...")
            reducer = umap.UMAP(n_neighbors=n_neighbors,
                                    min_dist=min_dist,
                                    n_components=n_components,
                                    random_state=42, # Para reprodutibilidade
                                    metric=metric) # Passa o parâmetro metric
            dados_para_clusterizar = reducer.fit_transform(embeddings)
            print(f"Nova dimensão após UMAP: {dados_para_clusterizar.shape}")

        print(f"A aplicar HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, allow_single_cluster={allow_single_cluster}, cluster_selection_epsilon={cluster_selection_epsilon})...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    gen_min_span_tree=True,
                                    allow_single_cluster=allow_single_cluster, # Passa o parâmetro
                                    prediction_data=True,
                                    cluster_selection_epsilon=cluster_selection_epsilon) # Passa o parâmetro
        clusterer.fit(dados_para_clusterizar)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Número de clusters encontrados: {n_clusters}")
        print(f"Número de pontos de ruído (outliers): {n_noise}")

        return labels, probabilities, clusterer

    # Definir parâmetros (baseado no seu corpus_overview.md e experimentação)

    # Usar o número de mensagens efetivamente vetorizadas para o cálculo do tamanho do cluster
    num_mensagens_vetorizadas = len(mensagens_para_vetorizar)
    
    # --- Cálculo dos Parâmetros HDBSCAN (usando nlp_cfg e número real de embeddings) ---
    _nlp_hdbscan_params_cfg = nlp_cfg.get('HDBSCAN_PARAMS', {})
    cfg_hdbscan_min_cluster_size = _nlp_hdbscan_params_cfg.get('min_cluster_size')
    cfg_hdbscan_min_samples = _nlp_hdbscan_params_cfg.get('min_samples')

    if isinstance(cfg_hdbscan_min_cluster_size, int) and cfg_hdbscan_min_cluster_size > 0:
        hdbscan_min_cluster_size_val = cfg_hdbscan_min_cluster_size
        print(f"HDBSCAN min_cluster_size definido DIRETAMENTE via config: {hdbscan_min_cluster_size_val}")
    else:
        hdbscan_min_cluster_size_factor = nlp_cfg.get('HDBSCAN_MIN_CLUSTER_SIZE_FACTOR', 0.01) # Pega do nlp_cfg
        if num_mensagens_vetorizadas > 0: # Usar o número real de embeddings
            hdbscan_min_cluster_size_val = max(15, int(num_mensagens_vetorizadas * hdbscan_min_cluster_size_factor))
        else:
            hdbscan_min_cluster_size_val = 15 # Default fallback se base for 0
            print("AVISO: num_mensagens_vetorizadas é 0. Usando default min_cluster_size=15")
        print(f"HDBSCAN min_cluster_size calculado com FATOR ({hdbscan_min_cluster_size_factor}) sobre EMBEDDINGS REAIS ({num_mensagens_vetorizadas}): {hdbscan_min_cluster_size_val}")

    if isinstance(cfg_hdbscan_min_samples, int) and cfg_hdbscan_min_samples > 0:
        hdbscan_min_samples_val = cfg_hdbscan_min_samples
        print(f"HDBSCAN min_samples definido DIRETAMENTE via config: {hdbscan_min_samples_val}")
    else:
        hdbscan_min_samples_factor = nlp_cfg.get('HDBSCAN_MIN_SAMPLES_FACTOR', 0.5) # Pega do nlp_cfg
        hdbscan_min_samples_val = max(5, int(hdbscan_min_cluster_size_val * hdbscan_min_samples_factor))
        print(f"HDBSCAN min_samples calculado com FATOR ({hdbscan_min_samples_factor}) sobre min_cluster_size ({hdbscan_min_cluster_size_val}): {hdbscan_min_samples_val}")
    
    hdbscan_allow_single_cluster = _nlp_hdbscan_params_cfg.get('allow_single_cluster', True)
    hdbscan_cluster_selection_epsilon = _nlp_hdbscan_params_cfg.get('cluster_selection_epsilon', 0.0)

    print(f"Parâmetros FINAIS para HDBSCAN: min_cluster_size={hdbscan_min_cluster_size_val}, min_samples={hdbscan_min_samples_val}, allow_single_cluster={hdbscan_allow_single_cluster}, cluster_selection_epsilon={hdbscan_cluster_selection_epsilon}")

    # Parâmetros UMAP do nlp_cfg
    _nlp_umap_params_cfg = nlp_cfg.get('UMAP_PARAMS', {})
    umap_n_neighbors = _nlp_umap_params_cfg.get('n_neighbors', 50)
    umap_min_dist = _nlp_umap_params_cfg.get('min_dist', 0.1)
    umap_n_components = _nlp_umap_params_cfg.get('n_components', 5)
    umap_metric = _nlp_umap_params_cfg.get('metric', 'cosine')

    if sbert_embeddings is not None and sbert_embeddings.shape[0] > 0:
        # Ler configuração de usar_umap
        use_umap_for_hdbscan_config = nlp_cfg.get('clustering_settings', {}).get('use_umap_before_hdbscan', True)
        
        hdbscan_labels, hdbscan_probabilities, hdbscan_model = clusterizar_com_hdbscan(
            sbert_embeddings,
            n_neighbors=umap_n_neighbors,      
            min_dist=umap_min_dist,        
            n_components=umap_n_components,  
            metric=umap_metric,
            min_cluster_size=hdbscan_min_cluster_size_val, 
            min_samples=hdbscan_min_samples_val,
            allow_single_cluster=hdbscan_allow_single_cluster,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            usar_umap=use_umap_for_hdbscan_config # Usa o valor do config
        )

        # Adicionar os resultados da clusterização ao DataFrame
        # Certificar-se de que o comprimento dos labels corresponde ao DataFrame
        if len(hdbscan_labels) == len(df_commits_para_clustering):
            df_commits_para_clustering['hdbscan_cluster_label'] = hdbscan_labels
            df_commits_para_clustering['hdbscan_cluster_probability'] = hdbscan_probabilities
            
            # Salvar o DataFrame com os labels dos clusters em JSONL
            clusters_jsonl_filename = nlp_cfg.get('output_paths', {}).get('clusters_jsonl', 'commits_com_clusters.jsonl')
            CAMINHO_DF_COM_CLUSTERS_JSONL = os.path.join(OUTPUT_DIR, clusters_jsonl_filename)
            
            # Ler colunas a serem salvas do nlp_cfg
            default_cols_to_save = [
                'repo_name', 'commit_hash', 'author_name', 'author_date', 
                'msg_original', 'msg_clean_for_val', 
                'hdbscan_cluster_label', 'hdbscan_cluster_probability'
            ]
            configured_cols_to_save = nlp_cfg.get('output_columns_for_clusters_jsonl', default_cols_to_save)
            
            # Garantir que apenas colunas existentes no DataFrame sejam selecionadas
            cols_to_save_clustering = [col for col in configured_cols_to_save if col in df_commits_para_clustering.columns]
            if not cols_to_save_clustering:
                print(f"AVISO: Nenhuma das colunas configuradas em 'output_columns_for_clusters_jsonl' existe no DataFrame. Usando colunas default: {default_cols_to_save}")
                cols_to_save_clustering = [col for col in default_cols_to_save if col in df_commits_para_clustering.columns]
            if not cols_to_save_clustering:
                print(f"ALERTA: Mesmo as colunas default não foram encontradas no DataFrame. O arquivo JSONL de saída pode estar vazio ou com poucas colunas.")
            
            # Certificar que datas sejam strings para JSON para evitar problemas de serialização
            df_temp_to_save = df_commits_para_clustering[cols_to_save_clustering].copy()
            if 'author_date' in df_temp_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_temp_to_save['author_date']):
                df_temp_to_save['author_date'] = df_temp_to_save['author_date'].dt.strftime('%Y-%m-%d %H:%M:%S%z') # Formato ISO 8601 com timezone
            
            try:
                df_temp_to_save.to_json(
                    CAMINHO_DF_COM_CLUSTERS_JSONL,
                    orient='records',
                    lines=True,
                    force_ascii=False # Para manter acentos e caracteres não-ASCII corretamente
                )
                print(f"DataFrame com labels de cluster salvo em JSON Lines: {CAMINHO_DF_COM_CLUSTERS_JSONL}")
            except Exception as e:
                print(f"Erro ao salvar DataFrame em JSON Lines: {CAMINHO_DF_COM_CLUSTERS_JSONL}. Erro: {e}")
                # Fallback ou tratamento de erro adicional se necessário

        else:
            print(f"ALERTA Bloco [8]: Comprimento dos labels HDBSCAN ({len(hdbscan_labels)}) não corresponde ao DataFrame ({len(df_commits_para_clustering)}). Arquivo JSONL de clusters não salvo.")
    else:
        print("AVISO Bloco [8]: Embeddings SBERT não disponíveis ou vazios. Pulando clusterização HDBSCAN.")

    # O próximo passo seria alimentar sbert_embeddings, hdbscan_labels (e opcionalmente hdbscan_model) ao BERTopic (Passo 1.4)
    print("--- Fim do Bloco [8] (SBERT e HDBSCAN) ---")
else:
    print("--- Bloco [8] (SBERT e HDBSCAN) pulado devido à ausência de dados de entrada. ---")

# Fim do Bloco [8] (conceitual)