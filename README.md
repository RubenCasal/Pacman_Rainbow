# Rainbow DQN for Ms Pacman

## Descripción
Este proyecto forma parte de un Trabajo de Fin de Grado y se centra en la implementación y estudio del algoritmo Rainbow DQN en el videojuego clásico Ms Pacman. Rainbow DQN es una versión avanzada del algoritmo DQN que incluye mejoras clave para optimizar el aprendizaje por refuerzo en entornos complejos como los juegos de Atari.

## Componentes del Algoritmo Rainbow DQN
- Double DQN (DDQN)
- Prioritized Experience Replay
- Dueling Networks
- Multi-step Learning
- Distributional DQN
- Noisy Networks
- N-step Learning

## Objetivo del Proyecto
El objetivo es realizar experimentaciones rigurosas para evaluar cómo cada componente impacta en el rendimiento del algoritmo. Se busca comprender a fondo el funcionamiento de Rainbow DQN en un entorno de juego desafiante y determinar la contribución de cada mejora al rendimiento general del sistema. Este estudio no solo ampliará el conocimiento sobre la eficacia de Rainbow DQN, sino que también proporcionará una valiosa exploración de métodos avanzados de aprendizaje automático aplicados en contextos de juegos complejos.

## Instalación
Para ejecutar este proyecto, debes preparar un entorno virtual utilizando Anaconda con Python 3.8. Sigue los siguientes pasos para configurar el entorno e instalar todas las dependencias necesarias:
1. **Abrir Visual Studio Code desde el navegador de Anaconda (Se debe tener instalado Anaconda)**

<img src="/ReadMe files/tutorial instalacion.png"  width="600">
2. **Clonar Repositorio**

   ```bash
      git clone https://github.com/RubenCasal/Pacman_Rainbow.git
    ```

1. **Crear un entorno virtual en Anaconda:**
   ```bash
        conda create -n myenv python=3.8
    ```
2. **Activa el entorno virtual en Anaconda:**
    ```bash
        conda activate myenv
    ```

3. **Instala las dependencias necesarias que se encuentran en el archivo requirements.txt:**
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
- `OPTIMIZER`: Tipo de optimizador que se utilizara para entrenar al modelo ('adam' | 'sgd' | 'rmsprop' | 'adamw')
Ejecuta el script desde la terminal con el siguiente comando:

```bash
python main.py
```
## Resultados
El rendimiento del modelo final alcanzado se puede ver en la siguiente gráfica, donde las líneas azules representan la puntuación alcanzada por el agente en un episodio determinado y la línea roja representa la media móvil de los últimos 100 episodios.
**insertar gráfica**
Tras diversas experimentaciones variando los distintos parámetros, hiperparámetros y arquitecturas del modelo, hemos identificado las configuraciones que ofrecen el mejor rendimiento para el algoritmo Rainbow DQN aplicado en el juego Ms Pacman:

- **Arquitectura**: 4 capas convolucionales seguidas de 4 capas lineales, y las 2 capas finales de actor y crítico.
- **N-step Learning**: Utilización de `n=2` para un enfoque de 2 pasos en el aprendizaje.
- **Prioritized Replay**: Uso de un buffer de replay con capacidad para 40,000 experiencias, permitiendo priorizar aquellas más relevantes para el aprendizaje.
- **Estrategias de Recompensas**: Implementación de recompensas normalizadas utilizando un logaritmo en base 1000 para escalar los valores de manera efectiva.
- **Noisy Networks**: Inclusión de ruido en todas las capas lineales con un `std=0.5` para promover la exploración.
- **Learning rate**: Se ha utilizado un valor de 0.0001 como learning rate para entrenar al modelo
- - **Optimizador**: Rellenar
<img src="/ReadMe files/pacman_best_model_log.png" alt="Descripción de la imagen" width="600">
<a href="/ReadMe files/log_rewards.mkv" download>Descargar el video</a>

**insertar video**



