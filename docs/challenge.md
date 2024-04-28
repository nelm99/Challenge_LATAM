# Principio de Trabajo

## Modelo de Machine Learning Robusto y Modular

En el desarrollo del archivo `model.py`, se aplicó un enfoque enfocado en la creación de un modelo de Machine Learning robusto y modular. Se implementaron las siguientes etapas clave:

### Verificación de Modelo Existente

El primer paso es verificar si ya existe un modelo previamente entrenado. Este enfoque es integral y considera diferentes escenarios posibles, permitiendo una mayor flexibilidad y reutilización del código.

### Preprocesamiento de Datos

Basándome en las exploraciones y análisis realizados en el archivo `exploration.ipynb`, se crearon funciones estáticas y métodos asociados a la clase `Delay` para el preprocesamiento de los datos. Esto asegura que los datos se preparan correctamente antes de cualquier operación de entrenamiento o predicción.

### Entrenamiento y Predicción

Tras el preprocesamiento, el siguiente paso es el entrenamiento del modelo seguido de la fase de predicción. Se ha mantenido la estructura lógica del archivo `.ipynb` original, pero se han simplificado ciertas partes del código para optimizar la eficiencia.

## Estructura del Proyecto

La estructura del proyecto se organiza en carpetas descriptivas, facilitando la navegación y la identificación de los archivos relevantes:

- `docs`: Contiene documentación relevante del proyecto.
- `data`: Incluye los datasets utilizados para el entrenamiento y las pruebas del modelo.
- `logs`: Almacena los registros de eventos y errores generados durante la ejecución del código.
- `model`: Contiene el modelo entrenado (en este caso, `challenge.json`) y otros archivos relacionados con el modelo de Machine Learning.
- `tests`: Incluye pruebas para asegurar que el código se ejecute correctamente y el modelo funcione como se espera.
- `utils`: Alberga funciones de utilidad y herramientas auxiliares para el proyecto.

Se ha seguido una nomenclatura coherente y descriptiva a lo largo del proyecto para mantener una alta legibilidad y mantenibilidad del código.
