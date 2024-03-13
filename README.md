0
## Instrucciones para ejecutar el caso práctico yoga_app.py


La aplicación `yoga_app.py` es un servidor web basado en Flask que se utiliza para predecir la postura de yoga en una imagen. 
La aplicación acepta una imagen a través de una solicitud POST, la procesa y utiliza un modelo de aprendizaje automático 
para predecir la postura de yoga en la imagen. La predicción, junto con una imagen de ejemplo de la postura de yoga 
predicha y la confianza de la predicción, se muestran en una página de resultados.

## Pasos

1. Crea un entorno virtual en este directorio: python3 -m virtualenv .
2. Activa el entorno virtual: source bin/activate
3. Instala las dependencias: pip3 install -r requirements.txt
4. Ejecuta el servidor en un terminal: python3 yoga_app.py
5. La carpeta incluida dataset/test contiene images para testear, igulamente la carpeta images tambien

NOTAS:
Se creo primero con Teachable Machine un modelo de posturas, pero lo genero en js y  generaba el modelo mas que en bin al final tuve que generarlo
con codigo el .pd y los json.Esta bajo la carpeta teachable_machine e hize esto para obtener del bin y del json el saved_model
	- pip install tensorflowjsc
	- cd teachable_machine
     - tensorflowjs_converter --input_format=tfjs_layers_model --output_format=keras model.json model.hdf5

Decide luego con LinearAi elegi el proyecto de posturas d eyoga con sus iamgenes qu ey venias, pero con las imagenes no me deja subir todas las que 
yo queria, y el modelo generado en la carpeta \flask_yoga\save_model\lineair
no me predecia bien , con otras imagens solo me daba un 0,21, ya que el set de entrena,iento er muy bajo
 