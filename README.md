# MVP 3 - Ciência de Dados: Classificação da Qualidade de Vinhos

**Discente:** Andre Luiz Marques Serrano  
**Data:** 28/08/2025  
**Curso:** Especialização em Ciência de Dados  


---

## 1. Definição do Problema

O objetivo deste projeto é desenvolver um Mínimo Produto Viável (MVP) para um problema de classificação, utilizando técnicas de machine learning. A tarefa consiste em prever a qualidade de vinhos tintos com base em suas características físico-químicas, um desafio relevante que pode auxiliar em processos de controle de qualidade e segmentação de mercado na indústria vinícola.

Para isso, utilizaremos o dataset "Wine Quality" do repositório da UCI. O problema será modelado como uma classificação binária, onde o objetivo é distinguir vinhos de alta qualidade dos demais.

### Hipótese

As características físico-químicas de um vinho (como teor alcoólico, acidez, pH, etc.) contêm informações suficientes para prever se ele será classificado como de alta qualidade.

### Dataset

- **Fonte:** UCI Machine Learning Repository
- **Link:** [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Atributos:** O dataset possui 11 atributos numéricos (features) e 1 variável alvo (a nota de qualidade)

### Configuração do Ambiente

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configurações de visualização para os gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Ambiente configurado e bibliotecas importadas com sucesso!")
```

---

## 2. Preparação dos Dados

Nesta etapa, realizamos as operações de carga, limpeza e transformação dos dados para prepará-los para a modelagem.

### Operações Realizadas

1. **Carga dos Dados:** O dataset é carregado diretamente de uma URL pública, garantindo a reprodutibilidade do notebook.

2. **Engenharia de Features:** A variável alvo `quality` (uma nota de 0 a 10) é transformada em uma variável categórica binária, `quality_cat`. Vinhos com nota maior ou igual a 7 são classificados como "bons" (classe 1), e os demais como "ruins" (classe 0). Esta simplificação torna o problema mais prático e o resultado do modelo mais interpretável.

3. **Separação em Treino e Teste:** O dataset é dividido em 80% para treino e 20% para teste. Utilizamos a amostragem estratificada para garantir que a proporção de vinhos bons e ruins seja a mesma em ambos os conjuntos, o que é crucial para datasets com classes desbalanceadas.

4. **Padronização (Scaling):** Criamos uma versão padronizada dos dados (média 0, desvio padrão 1). Embora modelos baseados em árvores (como o Random Forest) não necessitem desta etapa, ela é fundamental para modelos sensíveis à escala, como a Regressão Logística, que usaremos como baseline.

### Implementação

#### 2.1. Carga dos dados

```python
# O separador de colunas neste arquivo é o ponto e vírgula (;)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

print("--- Amostra do Dataset Original ---")
display(data.head())
```

#### 2.2. Engenharia de Features: Criando a variável alvo binária

```python
data['quality_cat'] = np.where(data['quality'] >= 7, 1, 0)

# Separando as features (X) e o alvo (y)
X = data.drop(['quality', 'quality_cat'], axis=1)
y = data['quality_cat']

print("\n--- Distribuição das Classes (0 = Ruim, 1 = Bom) ---")
print(y.value_counts(normalize=True))
sns.countplot(x=y)
plt.title('Distribuição das Classes de Qualidade')
plt.show()
```

#### 2.3. Separação em Treino e Teste (Estratificada)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nFormato dos dados de treino: {X_train.shape}")
print(f"Formato dos dados de teste: {X_test.shape}")
```

#### 2.4. Padronização dos Dados

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nDados padronizados com sucesso para uso em modelos sensíveis à escala.")
```

---

## 3. Modelagem e Treinamento

Nesta seção, construímos e treinamos os modelos de machine learning. A abordagem escolhida foi:

- **Modelo Baseline (Regressão Logística):** Um modelo simples e rápido que serve como um ponto de referência. Se um modelo mais complexo não superar significativamente este baseline, sua complexidade pode não ser justificada.

- **Modelo Principal (Random Forest):** Um algoritmo de ensemble robusto e de alta performance, que não exige padronização de dados e oferece insights sobre a importância das features.

- **Análise de Feature Importance:** Após treinar o Random Forest inicial, analisamos quais atributos mais influenciaram suas decisões. Isso nos ajuda a entender o "raciocínio" do modelo e a validar se ele está focando em características relevantes.

### Implementação

#### 3.1. Modelo Baseline: Regressão Logística

```python
# Usamos os dados escalados e 'class_weight' para lidar com o desbalanceamento
print("--- Treinando Modelo Baseline: Regressão Logística ---")
log_reg = LogisticRegression(random_state=42, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
print("Modelo de Regressão Logística treinado.")
```

#### 3.2. Modelo Principal: Random Forest Classifier

```python
# Não precisa de dados escalados. 'class_weight' também é usado aqui
print("\n--- Treinando Modelo Principal: Random Forest ---")
rf_initial = RandomForestClassifier(
    random_state=42, class_weight='balanced', n_estimators=100
)
rf_initial.fit(X_train, y_train)
print("Modelo Random Forest Inicial treinado.")
```

#### 3.3. Análise de Feature Importance

```python
print("\n--- Análise de Importância das Features (Random Forest) ---")
importances = rf_initial.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names, 
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Importância das Features para Prever a Qualidade do Vinho')
plt.xlabel('Importância Relativa')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

display(feature_importance_df)
```

---

## 4. Otimização de Hiperparâmetros

Um modelo de machine learning possui "botões" (hiperparâmetros) que controlam seu comportamento. Para extrair a máxima performance do Random Forest, utilizamos a técnica GridSearchCV, que realiza uma busca exaustiva pela melhor combinação desses hiperparâmetros.

### Justificativa

- **Validação Cruzada (cv=5):** O GridSearchCV utiliza validação cruzada para avaliar cada combinação de parâmetros. Isso fornece uma estimativa muito mais robusta da performance do modelo em dados não vistos, prevenindo a escolha de parâmetros que funcionam bem por mero acaso em uma única divisão de dados.

- **Métrica de Otimização (F1-Score):** Como o dataset é desbalanceado, otimizar apenas pela acurácia seria um erro. Escolhemos o f1_score ponderado, que representa uma média harmônica entre precisão e recall, fornecendo uma medida de performance muito mais equilibrada e confiável para este tipo de problema.

### Implementação

```python
print("--- Otimizando o Random Forest com GridSearchCV ---")

# Definindo a grade de parâmetros para testar
# Esta grade é menor para uma execução mais rápida como exemplo
# Em um projeto real, a grade pode ser mais ampla
param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurando o GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Usar todos os núcleos de CPU disponíveis
    scoring='f1_weighted',
    verbose=2
)

# Executando a busca
grid_search.fit(X_train, y_train)

print(f"\nMelhores hiperparâmetros encontrados: {grid_search.best_params_}")

# O melhor modelo já treinado com os melhores parâmetros
rf_optimized = grid_search.best_estimator_
```

---

## 5. Avaliação Final e Conclusão

Esta é a etapa final, onde avaliamos a performance dos nossos modelos no conjunto de teste — dados que eles nunca viram antes. Isso nos dá a melhor estimativa de como os modelos se comportariam em um cenário real.

### Análise dos Resultados

- **Comparação de Modelos:** Comparamos a performance (Acurácia e F1-Score) dos três modelos: o baseline, o Random Forest inicial e o Random Forest otimizado.

- **Análise de Overfitting:** Verificamos se o modelo final está "decorando" os dados de treino (overfitting) ou se ele generaliza bem para novos dados. Fazemos isso comparando sua pontuação no conjunto de treino com a do conjunto de teste.

- **Matriz de Confusão:** Visualizamos os erros e acertos do melhor modelo para entender que tipo de erros ele mais comete (ex: classificar um vinho bom como ruim, ou vice-versa).

- **Conclusão:** A melhor solução encontrada será o modelo com a maior performance no conjunto de teste, demonstrando que a abordagem de modelagem e otimização foi bem-sucedida.

### Implementação

```python
print("--- Avaliação Final dos Modelos no Conjunto de Teste ---")

# Fazendo previsões com todos os modelos
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_pred_rf_initial = rf_initial.predict(X_test)
y_pred_rf_optimized = rf_optimized.predict(X_test)
```

#### 5.1. Tabela Comparativa de Resultados

```python
results = {
    "Modelo": [
        "Regressão Logística (Baseline)",
        "Random Forest Inicial",
        "Random Forest Otimizado"
    ],
    "Acurácia": [
        accuracy_score(y_test, y_pred_log_reg),
        accuracy_score(y_test, y_pred_rf_initial),
        accuracy_score(y_test, y_pred_rf_optimized)
    ],
    "F1-Score (Ponderado)": [
        float(classification_report(y_test, y_pred_log_reg, output_dict=True)['weighted avg']['f1-score']),
        float(classification_report(y_test, y_pred_rf_initial, output_dict=True)['weighted avg']['f1-score']),
        float(classification_report(y_test, y_pred_rf_optimized, output_dict=True)['weighted avg']['f1-score'])
    ]
}

results_df = pd.DataFrame(results).set_index("Modelo")
print("\n--- Tabela Comparativa de Performance ---\n")
display(results_df.round(4))
```

#### 5.2. Relatório de Classificação Detalhado do Melhor Modelo

```python
print("\n--- Relatório de Classificação - Random Forest Otimizado ---")
print(classification_report(y_test, y_pred_rf_optimized, target_names=['Ruim (0)', 'Bom (1)']))
```

#### 5.3. Análise de Overfitting do Melhor Modelo

```python
train_score = rf_optimized.score(X_train, y_train)
test_score = rf_optimized.score(X_test, y_test)

print("\n--- Análise de Overfitting (Modelo Otimizado) ---")
print(f"Pontuação (Acurácia) no conjunto de Treino: {train_score:.


4f}")
print(f"Pontuação (Acurácia) no conjunto de Teste: {test_score:.4f}")

if train_score > test_score + 0.1:
    print("\nAlerta: Diferença significativa entre treino e teste. Pode haver overfitting.")
else:
    print("\nO modelo parece ter uma boa capacidade de generalização.")
```

#### 5.4. Matriz de Confusão do Melhor Modelo

```python
print("\n--- Matriz de Confusão (Modelo Otimizado) ---")
cm = confusion_matrix(y_test, y_pred_rf_optimized)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ruim', 'Bom'], 
            yticklabels=['Ruim', 'Bom'])
plt.xlabel('Previsão do Modelo')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão')
plt.show()
```

---

## Conclusões e Considerações Finais

Este terceiro MVP do curso de especialização demonstrou a aplicação prática de técnicas de machine learning para resolver um problema real de classificação na indústria vinícola. O projeto seguiu uma metodologia estruturada, desde a definição do problema até a avaliação final dos modelos.

### Principais Resultados

1. **Preparação de Dados Eficiente:** A transformação da variável alvo em uma classificação binária simplificou o problema e tornou os resultados mais interpretáveis para aplicações práticas.

2. **Comparação de Modelos:** A utilização de um modelo baseline (Regressão Logística) permitiu avaliar se a complexidade adicional do Random Forest era justificada.

3. **Otimização Sistemática:** O uso do GridSearchCV com validação cruzada garantiu uma seleção robusta de hiperparâmetros, maximizando a performance do modelo final.

4. **Avaliação Abrangente:** A análise incluiu múltiplas métricas (acurácia, F1-score), verificação de overfitting e visualização da matriz de confusão, proporcionando uma visão completa da performance do modelo.

### Aplicações Práticas

O modelo desenvolvido pode ser aplicado em:
- Controle de qualidade em vinícolas
- Segmentação de produtos para diferentes mercados
- Otimização de processos de produção
- Suporte à tomada de decisões comerciais

### Próximos Passos

Para trabalhos futuros, sugere-se:
- Exploração de outros algoritmos de machine learning
- Análise mais detalhada das features mais importantes
- Coleta de dados adicionais para melhorar a generalização
- Implementação de um sistema de monitoramento contínuo do modelo

---

**Desenvolvido por:** Andre Luiz Marques Serrano  
**Curso:** Especialização em Ciência de Dados  
**MVP:** 3/N  
**Data de Conclusão:** 23/08/2025

