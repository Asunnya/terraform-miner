# %% Importing libraries and configuration
import os
import sys
import re
import yaml
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Global variables for NLP models
nlp_en = None
lemmatizer_nltk_en = None

print("Libraries imported successfully!")

# %% Helper Functions
def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        print(f"CRITICAL: Configuration file not found at '{config_path}'")
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from '{config_path}'.")
        return config
    except Exception as e:
        print(f"CRITICAL: Error loading or parsing YAML configuration from '{config_path}': {e}")
        sys.exit(1)

# %% NLTK Resources Setup
def ensure_nltk_resources():
    """Ensures necessary NLTK resources are downloaded."""
    resources = {'corpora/stopwords': 'stopwords', 'tokenizers/punkt': 'punkt', 'corpora/wordnet': 'wordnet'}
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except Exception as e:
            print(f"Downloading NLTK resource: {package}...")
            nltk.download(package)

def load_spacy_model_en(use_spacy):
    """Loads spaCy model for English or falls back to NLTK."""
    nlp_en_model = None
    nltk_lemmatizer = None
    spacy_is_active = False

    if use_spacy:
        try:
            nlp_en_model = spacy.load('en_core_web_sm')
            print("spaCy 'en_core_web_sm' model loaded for English lemmatization.")
            spacy_is_active = True
        except IOError:
            print("WARNING: spaCy 'en_core_web_sm' model not found. Falling back to NLTK Lemmatizer.")
            spacy_is_active = False

    if not spacy_is_active:
        from nltk.stem import WordNetLemmatizer
        nltk_lemmatizer = WordNetLemmatizer()
        print("Using NLTK WordNetLemmatizer for English lemmatization.")
        
    return nlp_en_model, nltk_lemmatizer, spacy_is_active

# %% Stopwords Management
def load_stopwords_from_suggestions_yaml(suggestions_file_path):
    """Reads stopwords suggestions file and returns a set."""
    stopwords_set = set()
    if not suggestions_file_path or not os.path.exists(suggestions_file_path):
        print(f"INFO: Stopwords suggestions file not specified or not found: '{suggestions_file_path}'")
        return stopwords_set
    try:
        with open(suggestions_file_path, 'r', encoding='utf-8') as f:
            # Simple implementation: assuming one word per line under a specific key if needed
            # For this refactoring, we keep the original logic
            data = yaml.safe_load(f)
            custom_stopwords = data.get("custom_stopwords", [])
            for term in custom_stopwords:
                term = term.strip().lower()
                if ' ' in term:
                    stopwords_set.update(word for word in term.split() if len(word) > 1 and not word.isdigit())
                elif len(term) > 1 and not term.isdigit():
                    stopwords_set.add(term)
        print(f"Loaded {len(stopwords_set)} unique stopwords from file: {suggestions_file_path}")
    except Exception as e:
        print(f"ERROR reading stopwords suggestions file {suggestions_file_path}: {e}")
    return stopwords_set

def get_compiled_stopwords_en(spacy_nlp_model, config):
    """Compiles a set of stopwords for English text based on config."""
    compiled_stopwords = set()
    if spacy_nlp_model:
        compiled_stopwords.update(spacy_nlp_model.Defaults.stop_words)
    
    # Add manual terms from config if specified
    manual_terms = config.get('text_processing', {}).get('manual_stopwords', {})
    git_terms = manual_terms.get('git_commit_process_terms', [])
    not_useful_terms = manual_terms.get('not_useful_terms', [])
    
    compiled_stopwords.update(git_terms)
    compiled_stopwords.update(not_useful_terms)
    
    if config.get('text_processing', {}).get('use_dynamic_stopwords'):
        suggestions_path = config.get('paths', {}).get('stopwords_suggestions_file')
        dynamic_stopwords = load_stopwords_from_suggestions_yaml(suggestions_path)
        compiled_stopwords.update(dynamic_stopwords)
        
    return compiled_stopwords

# %% Text Preprocessing
def preprocess_text_for_keyword_extraction_en(text, spacy_en_model, nltk_en_lemmatizer, current_stop_words_set, use_spacy):
    """Cleans and lemmatizes text in ENGLISH for keyword extraction."""
    if text is None: return []
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(\s|^)(#|gh-|issue-|jira-)\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"signed-off-by:.*?<.*?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"co-authored-by:.*?<.*?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text, language='english')
    if not current_stop_words_set: current_stop_words_set = set()

    processed_tokens = []
    if spacy_en_model and use_spacy:
        doc = spacy_en_model(" ".join(tokens))
        processed_tokens = [
            token.lemma_ for token in doc
            if token.lemma_ not in current_stop_words_set and len(token.lemma_) > 1 and
            not token.is_punct and not token.is_space and
            not token.lemma_.isdigit() and not token.lemma_.startswith('-')
        ]
    elif nltk_en_lemmatizer:
        processed_tokens = [
            nltk_en_lemmatizer.lemmatize(word) for word in tokens
            if word not in current_stop_words_set and len(word) > 1 and not word.isdigit()
        ]
    else:
        processed_tokens = [
            word for word in tokens
            if word not in current_stop_words_set and len(word) > 1 and not word.isdigit()
        ]
    return processed_tokens

# %% Cluster Analysis and Reporting
def get_contextual_examples(search_series, display_series, term, config):
    """Finds, formats, and highlights examples of a term in messages."""
    text_cfg = config['text_processing']
    term_cfg = config['terms_analysis']
    
    # This regex finds the lemmatized term at the beginning of a word
    # and includes the rest of the word in the match, which is more robust.
    # e.g., for term 'conflict', it will match 'conflict', 'conflicts', 'conflicting'
    search_flags = re.IGNORECASE if text_cfg['case_insensitive_highlighting'] else 0
    # The pattern captures the entire word (base term + ending) for highlighting.
    regex_pattern = r'(\b' + re.escape(term) + r'[a-z]*)'

    examples = []
    
    # We iterate over the series to find matches and get the corresponding display message.
    # This is more explicit and correct than using a boolean mask from .str.contains().
    found_indices = []
    for index, message in search_series.items():
        if isinstance(message, str) and re.search(regex_pattern, message, flags=search_flags):
            found_indices.append(index)

    total_occurrences = len(found_indices)
    
    # Get the display messages for the found indices
    # Using .loc ensures we get the correct rows even if indices are not sequential
    containing_messages = display_series.loc[found_indices]

    # Limit to the number of examples we want to show
    for msg in containing_messages.head(term_cfg['examples_per_term']):
        formatted_msg = str(msg).strip()
        
        # Highlight the full word that was matched (e.g., **conflicts**)
        if text_cfg['highlight_terms_in_examples']:
            formatted_msg = re.sub(regex_pattern, r'**\1**', formatted_msg, flags=search_flags)
            
        # Truncate message
        if text_cfg['truncate_long_messages'] and len(formatted_msg) > term_cfg['max_example_length']:
            formatted_msg = formatted_msg[:term_cfg['max_example_length']] + "..."
            
        examples.append(formatted_msg)
        
    return examples, total_occurrences

def analyze_cluster_report(df_clustered, config, stop_words_to_use, nlp_models, spacy_is_active, output_report_path):
    """
    Analyzes each cluster to extract themes and generates a detailed report
    with contextual examples for top terms.
    """
    (current_nlp_en_model, current_nltk_lemmatizer) = nlp_models
    
    # Parameter mapping from config
    data_cfg = config['data_columns']
    term_cfg = config['terms_analysis']
    msg_cfg = config['message_samples']
    report_cfg = config['report_settings']
    section_cfg = config['report_sections']
    
    input_filename = os.path.basename(config['paths']['input_file_path'])
    
    if df_clustered.empty:
        print("Clustered DataFrame is empty. Skipping analysis.")
        return
        
    # Ensure report directory exists
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    
    df_clustered[data_cfg['label_column']] = df_clustered[data_cfg['label_column']].astype(int)
    unique_labels = sorted(df_clustered[data_cfg['label_column']].unique())
    
    with open(output_report_path, "w", encoding="utf-8") as f_report:
        # --- Report Header ---
        f_report.write("Semantic Analysis of Commit Message Clusters (HDBSCAN - English Focus)\n")
        f_report.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Input Data File: {input_filename}\n")
        f_report.write(f"Total clusters (including noise -1): {len(unique_labels)}\n")
        f_report.write(f"Total messages analyzed: {len(df_clustered)}\n")
        if report_cfg['include_stopwords_count']:
            f_report.write(f"Stopwords used (total): {len(stop_words_to_use)}\n")
        if report_cfg['include_lemmatization_info']:
            f_report.write(f"Lemmatization with spaCy: {'Yes' if spacy_is_active else 'No'}\n")
            f_report.write(f"Lemmatization with NLTK: {'Yes' if not spacy_is_active else 'No'}\n")
        f_report.write("===============================================================\n\n")

        # --- Per-Cluster Analysis ---
        for label_val in unique_labels:
            is_noise_cluster = (label_val == -1)
            cluster_name = "Noise (Outliers)" if is_noise_cluster else f"Cluster {label_val}"
            
            cluster_data = df_clustered[df_clustered[data_cfg['label_column']] == label_val]
            num_messages_in_cluster = len(cluster_data)
            
            if section_cfg['cluster_overview']:
                f_report.write(f"---------------------------------------------------------------\n")
                f_report.write(f"Cluster: {cluster_name}\n")
                percentage_str = f" ({num_messages_in_cluster/len(df_clustered):.2%})" if report_cfg['show_cluster_percentages'] else ""
                f_report.write(f"  Number of Messages: {num_messages_in_cluster}{percentage_str}\n\n")

            # Handling for noise cluster
            if is_noise_cluster:
                if section_cfg['message_samples']:
                    f_report.write(f"  These are considered outliers and do not form a cohesive theme.\n")
                    f_report.write(f"  Sample Messages (Max. {msg_cfg['noise_cluster_samples']}):\n")
                    sample_size = min(msg_cfg['noise_cluster_samples'], num_messages_in_cluster)
                    if sample_size > 0:
                        sample_noise = cluster_data.sample(sample_size, random_state=42)
                        for _, row in sample_noise.iterrows():
                            original_msg = str(row.get(data_cfg['display_message_column'], "N/A"))
                            f_report.write(f"    - \"{original_msg[:msg_cfg['max_sample_length']]}{'...' if len(original_msg) > msg_cfg['max_sample_length'] else ''}\"\n")
                f_report.write("\n")
                continue

            # --- Term Analysis for regular clusters ---
            # Pre-processing is always done on the search column to get the terms
            all_terms_in_cluster = []
            for msg_text in cluster_data[data_cfg['search_message_column']]:
                processed_terms = preprocess_text_for_keyword_extraction_en(
                    msg_text, current_nlp_en_model, current_nltk_lemmatizer, stop_words_to_use, spacy_is_active
                )
                all_terms_in_cluster.extend(processed_terms)
            
            term_counts = Counter(all_terms_in_cluster)
            top_terms = term_counts.most_common(term_cfg['num_top_terms'])
            
            if section_cfg['top_terms_list']:
                f_report.write(f"  TERMOS MAIS FREQUENTES (Top {term_cfg['num_top_terms']}):\n")
                if not top_terms:
                    f_report.write("    No significant terms found.\n")
                for term, count in top_terms:
                    f_report.write(f"    - {term}: {count}\n")
                f_report.write("\n")
            
            if section_cfg['detailed_term_contexts'] and top_terms:
                f_report.write("==============================================================\n")
                f_report.write(f"ANÁLISE CONTEXTUAL DOS TERMOS PRINCIPAIS ({term_cfg['num_terms_with_context']} exemplos por termo)\n")
                f_report.write("==============================================================\n\n")
                
                for i, (term, count) in enumerate(top_terms[:term_cfg['num_terms_with_context']]):
                    f_report.write(f"TERMO {i+1}: '{term}' ({count} ocorrências no cluster)\n")
                    f_report.write("--------------------------------------------------\n")
                    
                    examples, total_found = get_contextual_examples(
                        cluster_data[data_cfg['search_message_column']], 
                        cluster_data[data_cfg['display_message_column']], 
                        term, 
                        config
                    )
                    
                    if not examples:
                        f_report.write("  Nenhum exemplo de contexto encontrado para este termo.\n\n")
                        continue
                        
                    for ex_idx, example_msg in enumerate(examples):
                        f_report.write(f"  EXEMPLO {ex_idx + 1}: \"{example_msg}\"\n")

                    if report_cfg['show_additional_occurrences'] and total_found > len(examples):
                         f_report.write(f"  (... e mais {total_found - len(examples)} ocorrências ...)\n")
                    f_report.write("\n")

            if section_cfg['manual_analysis_placeholders']:
                f_report.write(f"  Hipótese de Tema do Cluster (Sua Interpretação):\n")
                f_report.write(f"    __________________________________________________________________\n\n")
                f_report.write(f"  Potencial Operador de Mutação Terraform Derivado (Sua Análise):\n")
                f_report.write(f"    __________________________________________________________________\n\n")

    print(f"Cluster analysis report saved to: {output_report_path}")

# %% Main Execution
def main():
    """Main function to orchestrate the analysis."""
    # --- 1. Load Configuration ---
    # Build a robust path to the config file, assuming it's in the same directory as the script.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd() # Fallback for interactive environments
        
    # Define project root as two levels up from the script's directory.
    project_root = os.path.dirname(os.path.dirname(script_dir))
        
    config_path = os.path.join(script_dir, 'config_02_analyze.yaml')
    config = load_config(config_path)
    
    # --- 2. Setup NLP Environment ---
    ensure_nltk_resources()
    use_spacy_from_config = config['text_processing'].get('use_spacy_lemmatization', True)
    nlp_en, nltk_lem, spacy_is_active = load_spacy_model_en(use_spacy=use_spacy_from_config)
    nlp_models = (nlp_en, nltk_lem)
    
    # --- 3. Load Data ---
    # Resolve the input file path relative to the project root for robustness.
    input_file_relative = config['paths']['input_file_path']
    input_file_absolute = os.path.join(project_root, input_file_relative)

    if not os.path.exists(input_file_absolute):
        print(f"ERROR: Input file not found: {input_file_absolute}")
        print("Please update the 'input_file_path' in 'config_02_analyze.yaml'.")
        return

    try:
        df_clustered_data = pd.read_json(input_file_absolute, lines=True)
        print(f"Loaded {len(df_clustered_data)} records from {input_file_absolute}")
        
        # Validate required columns
        required_cols = [
            config['data_columns']['label_column'], 
            config['data_columns']['search_message_column'],
            config['data_columns']['display_message_column']
        ]
        if not all(col in df_clustered_data.columns for col in required_cols):
            print(f"ERROR: One or more required columns not found in the data.")
            print(f"Required: {required_cols}")
            print(f"Available: {df_clustered_data.columns.tolist()}")
            return
            
    except Exception as e:
        print(f"ERROR loading input file '{input_file_absolute}': {e}")
        return

    # --- 4. Prepare Stopwords ---
    final_stopwords = get_compiled_stopwords_en(nlp_models[0], config)
    
    # --- 5. Define Output Path and Run Analysis ---
    # Resolve output directory relative to project root for consistency.
    output_dir_relative = config['paths']['output_dir']
    output_dir_absolute = os.path.join(project_root, output_dir_relative)

    report_subdir = "semantic_cluster_analysis_reports"
    full_output_dir = os.path.join(output_dir_absolute, report_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Create a dynamic filename based on config
    report_basename = f"cluster_analysis_report_{os.path.basename(input_file_relative).split('.')[0]}.txt"
    output_report_file_path = os.path.join(full_output_dir, report_basename)
    
    print(f"Output report will be saved to: {output_report_file_path}")
    
    # --- 6. Run Cluster Analysis ---
    analyze_cluster_report(
        df_clustered=df_clustered_data,
        config=config,
        stop_words_to_use=final_stopwords,
        nlp_models=nlp_models,
        spacy_is_active=spacy_is_active,
        output_report_path=output_report_file_path,
    )
    
    print(f"\n--- Analysis complete. Report saved to: {output_report_file_path} ---")

if __name__ == "__main__":
    main()