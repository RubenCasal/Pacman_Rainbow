# Rainbow DQN for Ms Pacman

## Descripción
Este proyecto forma parte de un Trabajo de Fin de Grado y se centra en la implementación y estudio del algoritmo Rainbow DQN en el videojuego clásico Ms Pacman. Rainbow DQN es una versión avanzada del algoritmo DQN que incluye mejoras clave para optimizar el aprendizaje por refuerzo en entornos complejos como los juegos de Atari.

## Componentes del Algoritmo Rainbow DQN

Rainbow DQN combina múltiples mejoras sobre el algoritmo básico DQN (Deep Q-Network), integrando características avanzadas que optimizan el rendimiento en entornos complejos. A continuación, se describen detalladamente los componentes que forman parte de Rainbow DQN:

### **1. Double DQN (DDQN)**

Double DQN aborda el problema de la sobreestimación de valores Q presente en el DQN estándar. En este caso, se utilizan dos redes diferentes:
- Una red principal para seleccionar la acción con el valor Q más alto.
- Una red objetivo para calcular el valor Q real de esa acción.

Esto reduce significativamente el sesgo en las estimaciones y mejora la estabilidad del entrenamiento. Double DQN mejora la precisión de las predicciones y evita decisiones subóptimas basadas en estimaciones sesgadas.


### **2. Prioritized Experience Replay**

Este componente asigna una prioridad a cada transición almacenada en el buffer de replay, seleccionando con mayor frecuencia las experiencias más relevantes. Las transiciones con mayores errores de predicción tienen prioridad, lo que acelera el aprendizaje y optimiza el uso de recursos computacionales.

Esto permite al agente centrarse en experiencias críticas que tienen un impacto significativo en el aprendizaje, mejorando la eficiencia general del algoritmo.


### **3. Dueling Networks**

Dueling Networks separa el cálculo del valor del estado y la ventaja de cada acción. La función Q se descompone en dos componentes:
- El valor del estado V(s), que representa lo "bueno" que es estar en un estado particular.
- La ventaja de la acción A(s, a), que indica la contribución específica de una acción en ese estado.

La función Q final se calcula como:
Q(s, a) = V(s) + A(s, a) - (1 / |A|) * Σ A(s, a')


Este enfoque mejora la capacidad del modelo para distinguir entre estados importantes y acciones irrelevantes, aumentando el rendimiento en entornos donde algunas acciones tienen poco impacto.


### **4. Multi-step Learning**

El aprendizaje multi-step extiende el enfoque estándar de 1-step TD Learning (Temporal Difference Learning) al utilizar múltiples pasos futuros para calcular las recompensas acumuladas. En lugar de actualizar los valores Q basándose únicamente en la recompensa inmediata, este componente utiliza la suma de recompensas acumuladas en los próximos *n* pasos.

Este enfoque captura información más rica sobre las recompensas futuras, proporcionando un balance más efectivo entre sesgo y varianza.


### **5. Distributional DQN**

Distributional DQN utiliza una representación probabilística de los valores Q, modelando la distribución completa de posibles valores Q en lugar de un único valor esperado. Esto permite al modelo capturar la incertidumbre inherente a cada acción y tomar decisiones más informadas.

La inclusión de distribuciones mejora la estabilidad del entrenamiento y permite una representación más robusta de las recompensas futuras, lo que contribuye a un mejor rendimiento general del agente.


### **6. Noisy Networks**

Las Noisy Networks introducen ruido en los pesos de las capas lineales de la red para fomentar la exploración durante el aprendizaje. Este enfoque sustituye las capas determinísticas por capas que incluyen ruido paramétrico, lo que permite al agente explorar de manera más eficiente.

La adición de ruido reduce la dependencia de estrategias de exploración como epsilon-greedy y promueve una exploración dirigida hacia regiones prometedoras del espacio de estados.


### **7. N-step Learning**

N-step Learning utiliza transiciones que incluyen n-pasos futuros en el cálculo de las recompensas acumuladas. Este enfoque combina información a corto y largo plazo, mejorando la calidad de las predicciones y la capacidad del modelo para capturar dependencias temporales complejas.


Cada uno de estos componentes aporta una mejora específica al algoritmo DQN original. Combinados, hacen de Rainbow DQN una solución robusta, eficiente y altamente efectiva para resolver problemas complejos en entornos como los juegos de Atari.


## Objetivo del Proyecto
El objetivo es realizar experimentaciones rigurosas para evaluar cómo cada componente impacta en el rendimiento del algoritmo. Se busca comprender a fondo el funcionamiento de Rainbow DQN en un entorno de juego desafiante y determinar la contribución de cada mejora al rendimiento general del sistema. Este estudio no solo ampliará el conocimiento sobre la eficacia de Rainbow DQN, sino que también proporcionará una valiosa exploración de métodos avanzados de aprendizaje automático aplicados en contextos de juegos complejos. La documentación de este proyecto se encuentra en el archivo llamado Aprendizaje por Refuerzo con Rainbow DQN en formato pdf [Aprendizaje por Refuerzo con Rainbow DQN](./ReadMe%20files/Aprendizaje%20por%20Refuerzo%20con%20Rainbow%20DQN.pdf).

## Instalación
Para ejecutar este proyecto, debes preparar un entorno virtual utilizando Anaconda con Python 3.8. Sigue los siguientes pasos para configurar el entorno e instalar todas las dependencias necesarias:
1. **Abrir Visual Studio Code desde el navegador de Anaconda (Se debe tener instalado Anaconda)**

<img src="/ReadMe files/tutorial instalacion.png"  width="600">

2. **Clonar Repositorio**

   ```bash
      git clone https://github.com/RubenCasal/Pacman_Rainbow.git
    ```

3. **Abrir consola CMD. Crear un entorno virtual en Anaconda:**
   ```bash
        conda create -n myenv python=3.8
    ```
4. **Activa el entorno virtual en Anaconda:**
    ```bash
        conda activate myenv
    ```

5. **Instala las dependencias necesarias que se encuentran en el archivo requirements.txt:**
     ```bash
    pip install -r requirements.txt
    ```

## Manual de Uso

Para utilizar el proyecto para entrenar o testear modelos, debes ejecutar el archivo `main.py`. Configura las siguientes variables en el código según tus necesidades:

- `device`: Define si usar 'cuda' para GPU o 'cpu' si no dispones de GPU compatible.
- `MODEL_PATH`: Especifica la ruta donde se cargará el modelo preentrenado en modo 'test'.
- `SAVE_MODEL_PATH`: Determina la ruta donde se guardará el modelo durante el entrenamiento.
- `MODE`: Establece 'test' para visualización gráfica y evaluación del rendimiento o 'train' para entrenar el modelo.
- `GRAPH_PATH`: Ruta donde se guardarán las gráficas de rendimiento del modelo.
- `LEARNING_RATE`: Valor que se le asignara como learning rate para entrenar al modelo.
- `EPISODES_TRAIN`: Número de episodios por los que se entrenara el modelo.
- `BUFFER_SIZE`: Cantidad de transiciones que será capaz de almacenar el buffer del Prioritized Experience Replay.
- `OPTIMIZER`: Tipo de optimizador que se utilizara para entrenar al modelo ('adam' | 'sgd' | 'rmsprop' | 'adamw')
Ejecuta el script desde la terminal con el siguiente comando:

```bash
python main.py
```
## Resultados
El rendimiento del modelo final alcanzado se puede ver en la siguiente gráfica, donde las líneas azules representan la puntuación alcanzada por el agente en un episodio determinado y la línea roja representa la media móvil de los últimos 100 episodios.

Tras diversas experimentaciones variando los distintos parámetros, hiperparámetros y arquitecturas del modelo, hemos identificado las configuraciones que ofrecen el mejor rendimiento para el algoritmo Rainbow DQN aplicado en el juego Ms Pacman:

- **Arquitectura**: 4 capas convolucionales seguidas de 4 capas lineales, y las 2 capas finales de actor y crítico.
- **N-step Learning**: Utilización de `n=2` para un enfoque de 2 pasos en el aprendizaje.
- **Prioritized Replay**: Uso de un buffer de replay con capacidad para 40,000 experiencias, permitiendo priorizar aquellas más relevantes para el aprendizaje.
- **Estrategias de Recompensas**: Implementación de recompensas normalizadas utilizando un logaritmo en base 1000 para escalar los valores de manera efectiva.
- **Noisy Networks**: Inclusión de ruido en todas las capas lineales con un `std=0.5` para promover la exploración.
- **Learning rate**: Se ha utilizado un valor de 0.0001 como learning rate para entrenar al modelo
- **Optimizador**: Se ha utilizado el optimizador Adam
- **Episodios**: Se ha entrenado durante 12.000 episodios
## Gráfica del entrenamiento
<img src="/ReadMe files/pacman_best_model_log.png" alt="Descripción de la imagen" width="600"> 

 **Pincha aquí para ver el video completo**

 
[![Demo Doccou alpha](https://github.com/RubenCasal/Pacman_Rainbow/blob/main/ReadMe%20files/gif%20mejorado.gif)](https://www.youtube.com/watch?v=H2qEoiEt10Q)


