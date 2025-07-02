import os
import logging
import time
import platform
import shutil
from git import Repo, GitCommandError

class RepoCloner:
    """
    Clonagem robusta de repos, tratando nomes longos, paths Windows e erros diversos.
    """
    def __init__(self, base_path='repos'):
        """
        Inicializa o clonador de repositórios.
        
        Parameters
        ----------
        base_path : str, optional
            Diretório base onde os repositórios serão clonados, por padrão 'repos'
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.is_windows = platform.system() == 'Windows'
        logging.info(f"RepoCloner inicializado. Diretório base: {self.base_path} (Sistema: {platform.system()})")
        
        # Log recomendação para Windows
        if self.is_windows:
            logging.info("Sistema Windows detectado. Recomendação: execute 'git config --global core.longpaths true'")

    def _sanitize_path(self, path):
        """
        Sanitiza o caminho para o sistema operacional atual,
        tratando caminhos longos no Windows.
        
        Parameters
        ----------
        path : str
            Caminho original
            
        Returns
        -------
        str
            Caminho sanitizado adequado para o SO atual
        """
        # Sempre use caminhos absolutos
        abs_path = os.path.abspath(path)
        
        # Para Windows, adicione o prefixo \\?\ para suporte a caminhos longos
        if self.is_windows and len(abs_path) > 250 and not abs_path.startswith('\\\\?\\'):
            return f"\\\\?\\{abs_path}"
        return abs_path

    def _safely_remove_directory(self, path):
        """
        Remove um diretório e seu conteúdo com segurança.
        
        Parameters
        ----------
        path : str
            Caminho do diretório a ser removido
            
        Returns
        -------
        bool
            True se bem-sucedido, False caso contrário
        """
        try:
            if os.path.exists(path):
                logging.info(f"Removendo diretório: {path}")
                
                if self.is_windows:
                    # No Windows, use rmtree com manipulador onerror para lidar com arquivos somente leitura
                    def remove_readonly(func, path, excinfo):
                        os.chmod(path, 0o777)
                        func(path)
                    
                    shutil.rmtree(path, onerror=remove_readonly)
                else:
                    shutil.rmtree(path)
                
                # Verificação dupla se a remoção foi bem-sucedida
                if os.path.exists(path):
                    logging.warning(f"Diretório ainda existe após tentativa de remoção: {path}")
                    return False
                return True
            return True  # Caminho não existe, então não é necessário remover
        except Exception as e:
            logging.error(f"Erro ao remover diretório {path}: {str(e)}")
            return False

    def clone(self, full_name, depth=None, partial_clone=True, retry_count=3, retry_delay=2):
        """
        Clona um repositório GitHub de forma robusta, com tratamento de erros e retentativas.
        
        Parameters
        ----------
        full_name : str
            Nome completo do repositório no formato 'owner/repo'
        depth : int, optional
            Profundidade do clone (para clones rasos), por padrão None (clone completo)
        partial_clone : bool, optional
            Se True, usa filtros de clone parcial para reduzir o tamanho dos dados, por padrão True
        retry_count : int, optional
            Número máximo de tentativas em caso de falha, por padrão 3
        retry_delay : int, optional
            Tempo de espera (em segundos) entre tentativas, por padrão 2
            
        Returns
        -------
        str
            Caminho para o repositório clonado
            
        Raises
        ------
        ValueError
            Se o formato do nome do repositório for inválido
        RuntimeError
            Se todas as tentativas de clonagem falharem
        """
        log_prefix = f"[RepoCloner][{full_name}]"
        logging.info(f"{log_prefix} Iniciando clonagem do repositório")
        
        # Separação do nome do proprietário e nome do repositório
        try:
            if '/' not in full_name:
                raise ValueError(f"Formato de nome de repositório inválido: {full_name}. Use o formato 'owner/repo'.")
                
            owner, name = full_name.split('/')
            
            # Usar nome de diretório mais curto com hash para garantir unicidade
            import hashlib
            repo_hash = hashlib.md5(full_name.encode()).hexdigest()[:8]
            safe_name = f"{owner}_{name[:50]}_{repo_hash}"  # Limita a owner + primeiros 50 chars do repo + hash
            
            dest = os.path.join(self.base_path, safe_name)
            dest = self._sanitize_path(dest)
            
            logging.info(f"{log_prefix} Caminho de destino: {dest}")
            
            # Verifica e limpa o diretório de destino se necessário
            if os.path.exists(dest) and os.listdir(dest):
                logging.info(f"{log_prefix} Diretório de destino existe e não está vazio.")
                try:
                    # Tenta verificar se é um repositório Git válido
                    repo = Repo(dest)
                    # Verifica se a origem é a correta
                    remote_url = next(repo.remotes.origin.urls)
                    expected_url = f"https://github.com/{full_name}.git"
                    
                    if expected_url.lower() not in remote_url.lower():
                        logging.warning(f"{log_prefix} URL remota incorreta. Removendo para reclonagem.")
                        if not self._safely_remove_directory(dest):
                            raise RuntimeError(f"Não foi possível remover diretório com URL remota incorreta.")
                    else:
                        # Repositório correto, tenta atualizar
                        logging.info(f"{log_prefix} Atualizando repositório existente...")
                        try:
                            repo.remotes.origin.fetch()
                            logging.info(f"{log_prefix} Repositório atualizado com sucesso.")
                            return dest
                        except Exception as e:
                            logging.warning(f"{log_prefix} Erro ao atualizar: {e}. Removendo para reclonagem.")
                            if not self._safely_remove_directory(dest):
                                raise RuntimeError(f"Não foi possível remover diretório após erro de atualização.")
                except Exception:
                    # Não é um repositório Git válido ou ocorreu outro erro
                    logging.warning(f"{log_prefix} Diretório existente inválido. Removendo para clonagem limpa.")
                    if not self._safely_remove_directory(dest):
                        raise RuntimeError(f"Não foi possível remover diretório inválido.")
            
            # Agora o diretório deve estar limpo ou não existir
            # Cria o diretório pai se necessário
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            # Clone do repositório com retentativas
            for attempt in range(1, retry_count + 1):
                try:
                    logging.info(f"{log_prefix} Tentativa {attempt}/{retry_count} - Clonando...")
                    
                    # Verificação final se o diretório está limpo
                    if os.path.exists(dest) and os.listdir(dest):
                        logging.warning(f"{log_prefix} Destino ainda não está limpo. Tentando novamente...")
                        if not self._safely_remove_directory(dest):
                            raise RuntimeError(f"Falha ao limpar diretório na tentativa {attempt}.")
                    
                    clone_url = f"https://github.com/{full_name}.git"
                    
                    # Opções de clone
                    clone_opts = {}
                    env = {}
                    
                    # Suporte a caminhos longos no Windows via variável de ambiente
                    if self.is_windows:
                        env = {'GIT_LFS_SKIP_SMUDGE': '1'}  # Também desativa LFS inicialmente
                        
                    # Opções de clone parcial
                    if partial_clone:
                        clone_opts['multi_options'] = ['--filter=blob:none']
                        logging.info(f"{log_prefix} Usando clone parcial para reduzir tamanho")
                    
                    # Opção de profundidade
                    if depth is not None and isinstance(depth, int) and depth > 0:
                        clone_opts['depth'] = depth
                        logging.info(f"{log_prefix} Usando clone raso com profundidade {depth}")
                    
                    # Executa a clonagem
                    start_time = time.time()
                    Repo.clone_from(clone_url, dest, env=env, **clone_opts)
                    elapsed = time.time() - start_time
                    
                    logging.info(f"{log_prefix} Clonagem bem-sucedida em {elapsed:.2f}s")
                    return dest
                    
                except GitCommandError as e:
                    error_msg = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                    logging.error(f"{log_prefix} Tentativa {attempt}/{retry_count} falhou: {error_msg}")
                    
                    # Tratamento de erros específicos
                    if "Filename too long" in error_msg:
                        if self.is_windows:
                            logging.error(f"{log_prefix} Erro de caminho muito longo no Windows. "
                                        "Execute 'git config --global core.longpaths true' e tente novamente.")
                    
                    if "already exists and is not an empty directory" in error_msg:
                        logging.warning(f"{log_prefix} Diretório não vazio. Tentando limpar...")
                        if not self._safely_remove_directory(dest):
                            logging.error(f"{log_prefix} Falha ao limpar diretório de destino.")
                            raise RuntimeError(f"Não foi possível limpar o diretório: {dest}")
                    
                    if attempt < retry_count:
                        wait_time = retry_delay * attempt  # Tempo de espera progressivo
                        logging.info(f"{log_prefix} Aguardando {wait_time}s antes da próxima tentativa...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"{log_prefix} Todas as tentativas de clonagem falharam.")
                        raise RuntimeError(f"Falha ao clonar {full_name} após {retry_count} tentativas: {error_msg}")
                
                except Exception as e:
                    logging.exception(f"{log_prefix} Erro inesperado: {str(e)}")
                    if attempt < retry_count:
                        wait_time = retry_delay * attempt
                        logging.info(f"{log_prefix} Aguardando {wait_time}s antes da próxima tentativa...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Erro inesperado ao clonar {full_name}: {str(e)}")
        
        except ValueError as ve:
            logging.error(f"{log_prefix} {str(ve)}")
            raise ve
        
        except Exception as e:
            logging.exception(f"{log_prefix} Erro não tratado: {str(e)}")
            raise RuntimeError(f"Erro não tratado ao clonar {full_name}: {str(e)}")