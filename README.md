# Clasificador de Imágenes de Fórmula 1

Este proyecto es un estudio comparativo exhaustivo sobre la clasificación de imágenes de Fórmula 1, implementando y evaluando arquitecturas de Deep Learning en los frameworks Keras y PyTorch.

---

## Descripción del Proyecto

- Este proyecto presenta el desarrollo, la optimización y la evaluación de cuatro modelos de clasificación de imágenes para distinguir entre tres clases del mundo de la F1: `crash`, `f1` (monoplaza) y `safety_car`.
- Se implementaron múltiples arquitecturas, desde una Red Neuronal Convolucional (CNN) con Transfer Learning hasta un modelo Híbrido que combina una CNN con un Vision Transformer (ViT).
- El objetivo es realizar un análisis comparativo riguroso, no solo entre las arquitecturas, sino también entre los flujos de trabajo de Keras y PyTorch, para determinar la solución más efectiva para este problema específico.

## Análisis y Metodología

### El proyecto se estructura en las siguientes fases:

1.  **Análisis Exploratorio de Datos (EDA) y Preprocesamiento:**
    - Se utilizó `tf.data` (Keras) y `ImageFolder` (PyTorch) para crear pipelines de datos eficientes.
    - Se aplicó **Data Augmentation** (giros, rotaciones, zoom) para enriquecer el dataset de entrenamiento y mitigar el sobreajuste.
    - Se analizaron las distribuciones de clases mediante gráficos de barras para confirmar el ligero desequilibrio del dataset.

2.  **Modelo 1 y 2: CNN con Transfer Learning**
    - **Técnica:** Se utilizó la técnica de **Transfer Learning**, empleando el modelo `EfficientNetB0` (pre-entrenado en ImageNet) como extractor de características. Se congelaron las capas base y se entrenó un cabezal de clasificación personalizado.
    - **Optimización:** Se realizó una búsqueda sistemática de hiperparámetros.
      - En **Keras**, se utilizó **Keras Tuner** para optimizar la tasa de aprendizaje y el dropout.
      - En **PyTorch**, se utilizó **Optuna** con el mismo objetivo.
    - **Resultado:** Dos modelos CNN optimizados, uno por cada framework, que sirven como una sólida línea base de rendimiento.

3.  **Modelo 3 y 4: Sistema Híbrido (CNN-ViT)**
    - **Técnica:** Se construyó una arquitectura híbrida más avanzada.
      - Se utilizó el modelo **CNN optimizado** de la fase anterior como un "backbone" o extractor de características ya especializado.
      - A la salida de la CNN se le conectó un **Vision Transformer (ViT)**, implementado con capas personalizadas, para que analizara las relaciones globales entre las características extraídas.
    - **Optimización:** Nuevamente, se utilizaron **Keras Tuner** y **Optuna** para optimizar no solo la tasa de aprendizaje, sino también los hiperparámetros de la arquitectura del Transformer (ej. número de capas).
    - **Resultado:** Dos modelos híbridos optimizados que evalúan si la complejidad añadida del Transformer se traduce en una mejora del rendimiento.

4.  **Comparación Final de Modelos**
    - Se consolidaron las métricas de rendimiento clave (Accuracy, F1-Score Macro y Ponderado) de los cuatro modelos finales en una tabla comparativa.
    - Se generaron gráficos para visualizar las diferencias de rendimiento y se redactó una conclusión basada en los datos para determinar la mejor combinación de arquitectura y framework.

---

## Tecnologías Utilizadas

- **Python 3.10**
- **Pandas & NumPy:** Para manipulación de datos.
- **Matplotlib & Seaborn:** Para visualización de datos.
- **Scikit-learn:** Para métricas de evaluación detalladas (Reporte de Clasificación, Matriz de Confusión).
- **TensorFlow & Keras:** Para la construcción, entrenamiento y optimización de los modelos en Keras.
- **PyTorch:** Para la construcción, entrenamiento y optimización de los modelos en PyTorch.
- **Keras Tuner & Optuna:** Para la optimización automática de hiperparámetros.
- **tensorflow-directml & torch-directml:** Para habilitar la aceleración por GPU en hardware AMD.

---

## Instrucciones de Instalación

1.  **Clona este repositorio**
    ```bash
    git clone https://github.com/geronimo290/Clasificador-de-Im-genes-Deep-Learning-con-Keras-y-Pytorch-

    cd Clasificador-de-Im-genes-Deep-Learning-con-Keras-y-Pytorch-
    
    ```

2.  **Crea un entorno virtual (recomendado con Conda)**
    ```bash
    conda create -n f1_classifier python=3.10
    conda activate f1_classifier
    ```

3.  **Instala las dependencias**

    **Opción A (Recomendada): Usando `requirements.txt`**
    
    Crea un archivo `requirements.txt` con el siguiente contenido y luego ejecuta `pip install -r requirements.txt`.
    ```txt
    # Frameworks de Deep Learning (con soporte para GPU AMD en Windows)
    tensorflow-directml
    torch-directml

    # Optimización y Métricas
    keras-tuner
    optuna
    scikit-learn

    # Análisis de Datos y Visualización
    pandas
    matplotlib
    seaborn

    # Entorno de Notebook
    notebook
    ```
    ```bash
    pip install -r requirements.txt
    ```

    **Opción B (Alternativa): Comandos Individuales**
    ```bash
    pip install tensorflow-directml torch-directml keras-tuner optuna scikit-learn pandas matplotlib seaborn notebook
    ```

4.  **Ejecuta los notebooks**
    Abre y ejecuta los notebooks con Jupyter Notebook en el entorno activado:
    ```bash
    jupyter notebook
    ```
