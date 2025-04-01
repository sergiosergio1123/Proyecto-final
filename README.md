from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cambia el nombre del modelo a "t5-small" o "t5-base"
nombre_modelo = "t5-small"  # O "t5-base"

# Importar los módulos necesarios del modelo T5
convertir_vectores = T5Tokenizer.from_pretrained(nombre_modelo, legacy=False)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

# Diccionario para mapear opciones con prefijos de tarea
prefijos_tarea = {
    1: "summarize: ",
    2: "translate: ",
    3: "question: ",
    4: "generate questions: "  # Nueva opción
}

def obtener_lenguajes_traduccion():
    lenguaje_origen = input("Ingresa el lenguaje de origen (ej: English, Spanish): ")
    lenguaje_destino = input("Ingresa el lenguaje de destino (ej: French, German): ")
    return lenguaje_origen, lenguaje_destino

# Mostrar un menú para que el usuario elija la opción que desea realizar
print("¿Qué tarea quieres realizar?")
print("1. Resumir")
print("2. Traducir")
print("3. Hacer una pregunta")
print("4. Generar preguntas")  # Nueva opción

# Leer la elección del usuario con manejo de errores
while True:
    try:
        eleccion = int(input("Ingresa el número de la tarea: "))
        if eleccion in prefijos_tarea:
            break
        else:
            print("Opción no válida. Ingresa 1, 2, 3 o 4.")
    except ValueError:
        print("Entrada no válida. Ingresa un número.")

# Variable para evaluar qué orden se le dará al modelo
tipo_tarea = prefijos_tarea[eleccion]

# Solicitar el texto que se quiere procesar
texto = input("Ingresa el texto: ")

print(f"Texto de entrada: {texto}")  # Agregar declaración print()

if eleccion == 2:
    lenguaje_origen, lenguaje_destino = obtener_lenguajes_traduccion()
    tipo_tarea = f"{tipo_tarea} {lenguaje_origen} to {lenguaje_destino}: "
    texto = f"{tipo_tarea} {texto}"
elif eleccion == 3:
    contexto = input("Ingresa el contexto: ")
    texto = f"{tipo_tarea} {texto} context: {contexto}"
elif eleccion == 4:  # Nueva opción
    print("\nRESULTADO")
    print(f"Tarea: {tipo_tarea}")
    for _ in range(10):
        # Calcular los vectores de entrada para generar una pregunta
        vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids

        # Calcular los vectores de salida para generar una pregunta
        vectores_salida = modelo.generate(vectores_entrada, max_length=100)  # Ajustar max_length

        # Decodificar los vectores de salida para generar la pregunta
        pregunta = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

        print(pregunta)  # Imprimir la pregunta generada
else:
    # Calcular los vectores de entrada
    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids

    # Calcular los vectores de salida
    vectores_salida = modelo.generate(vectores_entrada, max_length=150)

    # Decodificar los vectores de salida para generar el resultado
    texto_salida = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

    # Mostrar el texto resumido
    print("\nRESULTADO")
    print(f"Tarea: {tipo_tarea}")
    print(texto_salida)
