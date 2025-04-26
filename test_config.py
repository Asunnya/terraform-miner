from src.analysis.config import get_config, get_analysis_summary

def main():
    config = get_config()
    print('Configuração carregada:', config)
    
    summary = get_analysis_summary(config['output_dir'])
    print('\nSumário de análise:', summary)

if __name__ == '__main__':
    main() 