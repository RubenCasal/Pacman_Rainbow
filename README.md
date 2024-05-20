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

Ejecuta el script desde la terminal con el siguiente comando:

```bash
python main.py
```
  


