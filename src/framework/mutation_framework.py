# mutation_framework.py

import os
import json
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

class TerraformMutationOperator(ABC):
    """Classe base abstrata para todos os operadores de mutação em Terraform."""
    
    def __init__(self, 
                 operator_id: str, 
                 name: str, 
                 description: str, 
                 target_resources: List[str] = None,
                 examples: List[Dict] = None,
                 detected_in_analysis: bool = False,
                 frequency: float = 0.0,
                 confidence: float = 0.0):
        """
        Inicialização de um operador de mutação.
        
        Args:
            operator_id: Identificador único do operador
            name: Nome descritivo do operador
            description: Descrição detalhada do operador e seu propósito
            target_resources: Lista de tipos de recursos que este operador pode modificar
            examples: Lista de exemplos de código original e mutado
            detected_in_analysis: Flag indicando se foi detectado na análise automática
            frequency: Frequência relativa de ocorrência na análise (0.0-1.0)
            confidence: Nível de confiança na detecção (0.0-1.0)
        """
        self.operator_id = operator_id
        self.name = name
        self.description = description
        self.target_resources = target_resources or ["*"]
        self.examples = examples or []
        self.detected_in_analysis = detected_in_analysis
        self.frequency = frequency
        self.confidence = confidence
        self.related_problems = []
        
    def add_example(self, original_code: str, mutated_code: str, description: str) -> None:
        """Adiciona um exemplo para este operador de mutação."""
        self.examples.append({
            "original": original_code,
            "mutated": mutated_code,
            "description": description
        })
    
    def add_related_problem(self, problem_description: str, severity: str = "medium") -> None:
        """Associa este operador a um problema específico encontrado na análise."""
        self.related_problems.append({
            "description": problem_description,
            "severity": severity
        })
    
    @abstractmethod
    def apply(self, terraform_code: str) -> str:
        """
        Aplica a mutação ao código Terraform.
        
        Args:
            terraform_code: Código Terraform como string
            
        Returns:
            Código Terraform mutado
        """
        pass
    
    def to_dict(self) -> Dict:
        """Converte o operador para um dicionário para serialização."""
        return {
            "operator_id": self.operator_id,
            "name": self.name,
            "description": self.description,
            "target_resources": self.target_resources,
            "examples": self.examples,
            "detected_in_analysis": self.detected_in_analysis,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "related_problems": self.related_problems
        }

class MutationCatalog:
    """Catálogo de operadores de mutação para Terraform."""
    
    def __init__(self, catalog_path: str = None):
        """
        Inicializa o catálogo de operadores de mutação.
        
        Args:
            catalog_path: Caminho para salvar/carregar o catálogo
        """
        self.operators = {}
        self.catalog_path = catalog_path or "terraform_mutation_catalog.json"
        
    def register_operator(self, operator: TerraformMutationOperator) -> None:
        """Registra um novo operador no catálogo."""
        if operator.operator_id in self.operators:
            print(f"Aviso: Substituindo operador existente com ID {operator.operator_id}")
        self.operators[operator.operator_id] = operator
        
    def get_operator(self, operator_id: str) -> Optional[TerraformMutationOperator]:
        """Recupera um operador pelo ID."""
        return self.operators.get(operator_id)
    
    def list_operators(self) -> List[TerraformMutationOperator]:
        """Lista todos os operadores registrados."""
        return list(self.operators.values())
    
    def save(self, path: str = None) -> None:
        """Salva o catálogo no formato JSON."""
        save_path = path or self.catalog_path
        
        # Converte todos os operadores para dicionários
        catalog_data = {
            "operators": {op_id: op.to_dict() for op_id, op in self.operators.items()}
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(catalog_data, f, indent=2, ensure_ascii=False)
        
        print(f"Catálogo salvo em {save_path}")
    
    def load(self, path: str = None) -> None:
        """
        Carrega o catálogo a partir de um arquivo JSON.
        Note: Esta implementação apenas carrega os metadados, não as implementações.
        """
        load_path = path or self.catalog_path
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                catalog_data = json.load(f)
                
            # Limpa operadores existentes
            self.operators = {}
            
            # Para uma implementação completa, você precisaria mapear os IDs dos operadores
            # para suas classes concretas. Esta é uma implementação simplificada.
            print(f"Catálogo carregado de {load_path} (apenas metadados)")
            print(f"Encontrados {len(catalog_data.get('operators', {}))} operadores")
        except FileNotFoundError:
            print(f"Arquivo de catálogo não encontrado em {load_path}")
    
    def analyze_diff_data(self, diff_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa dados de diff para identificar padrões potenciais para operadores de mutação.
        
        Args:
            diff_data: DataFrame contendo dados de diff/mudanças em código Terraform
            
        Returns:
            Dicionário com estatísticas e padrões identificados
        """
        analysis_results = {
            "resource_frequencies": {},
            "attribute_frequencies": {},
            "modification_patterns": [],
            "potential_operators": []
        }
        
        # Implementação da análise...
        # Esta é uma implementação esqueleto que você completaria com sua análise específica
        
        if 'resource_type' in diff_data.columns:
            analysis_results["resource_frequencies"] = diff_data['resource_type'].value_counts().to_dict()
        
        if 'attr_name' in diff_data.columns and 'resource_type' in diff_data.columns:
            # Análise de atributos por tipo de recurso
            for resource in diff_data['resource_type'].unique():
                if pd.notna(resource):
                    resource_df = diff_data[diff_data['resource_type'] == resource]
                    if 'attr_name' in resource_df.columns:
                        attr_counts = resource_df['attr_name'].value_counts()
                        if not attr_counts.empty:
                            analysis_results["attribute_frequencies"][resource] = attr_counts.to_dict()
        
        return analysis_results

class AnalysisDrivenCatalog(MutationCatalog):
    """Extensão do catálogo que usa análise de dados para criar operadores."""
    
    def generate_operators_from_analysis(self, diff_data: pd.DataFrame) -> List[str]:
        """
        Analisa dados de diff e gera operadores de mutação automaticamente.
        
        Args:
            diff_data: DataFrame contendo dados de mudanças em Terraform
            
        Returns:
            Lista de IDs dos operadores gerados
        """
        analysis = self.analyze_diff_data(diff_data)
        generated_ids = []
        
        # Exemplo: Gerar operadores baseados em atributos frequentemente modificados
        if analysis["attribute_frequencies"]:
            for resource, attrs in analysis["attribute_frequencies"].items():
                # Pegar os 3 atributos mais frequentes para cada recurso
                top_attrs = sorted(attrs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for attr_name, frequency in top_attrs:
                    if pd.notna(attr_name) and attr_name:
                        # Criar ID único para o operador
                        op_id = f"attr_change_{resource}_{attr_name}".replace(".", "_").lower()
                        
                        # Registrar um operador genérico para este atributo
                        # Na implementação real, você usaria uma subclasse adequada
                        class GenericAttributeMutator(TerraformMutationOperator):
                            def apply(self, terraform_code):
                                # Implementação de exemplo
                                return terraform_code
                        
                        # Criar instância do operador
                        operator = GenericAttributeMutator(
                            operator_id=op_id,
                            name=f"Mutação de {attr_name} em {resource}",
                            description=f"Modifica o atributo {attr_name} em recursos do tipo {resource}",
                            target_resources=[resource],
                            detected_in_analysis=True,
                            frequency=frequency / sum(attrs.values()) if sum(attrs.values()) > 0 else 0
                        )
                        
                        # Registrar o operador
                        self.register_operator(operator)
                        generated_ids.append(op_id)
        
        return generated_ids

class MutationOperatorFactory:
    """Fábrica para criar instâncias concretas de operadores de mutação."""
    
    @staticmethod
    def create_operator(operator_type: str, **kwargs) -> TerraformMutationOperator:
        """
        Cria uma instância de um operador de mutação específico.
        
        Args:
            operator_type: Tipo do operador a ser criado
            **kwargs: Argumentos para o construtor do operador
            
        Returns:
            Instância do operador de mutação
        """
        operator_classes = {

        }
        
        if operator_type not in operator_classes:
            raise ValueError(f"Tipo de operador desconhecido: {operator_type}")
        
        return operator_classes[operator_type](**kwargs)