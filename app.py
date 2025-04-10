from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Cargar modelo T5
nombre_modelo = "t5-small"
convertir_vectores = T5Tokenizer.from_pretrained(nombre_modelo, legacy=False)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

# Diccionario para mapear opciones con prefijos de tarea
prefijos_tarea = {
    1: "summarize: ",
    2: "translate: ",
    3: "question: ",
    4: "generate questions: "
}

def generar_preguntas(texto):
    """Genera 10 preguntas a partir de un texto."""
    preguntas = []
    fragmentos = [s.strip() for s in texto.split('.') if s.strip()]
    for fragmento in fragmentos[:10]:
        instrucciones = f"Genera una pregunta sobre el siguiente texto: '{fragmento}'."
        vectores_entrada = convertir_vectores(instrucciones, return_tensors="pt").input_ids
        vectores_salida = modelo.generate(vectores_entrada, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        pregunta = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True).strip()
        if not pregunta.endswith('?'):
            pregunta += '?'
        preguntas.append(pregunta)
    return preguntas

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        texto = request.form["texto"]
        eleccion = int(request.form["tarea"])

        tipo_tarea = prefijos_tarea.get(eleccion, "Opción no válida.")

        if eleccion == 1:
            instrucciones = f"{tipo_tarea} {texto}"
            vectores_entrada = convertir_vectores(instrucciones, return_tensors="pt").input_ids
            vectores_salida = modelo.generate(vectores_entrada, max_length=150)
            resultado = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)
        elif eleccion == 2:
            idioma_origen = request.form["idioma_origen"]
            idioma_destino = request.form["idioma_destino"]
            instrucciones = f"{tipo_tarea} {idioma_origen} to {idioma_destino}: {texto}"
            vectores_entrada = convertir_vectores(instrucciones, return_tensors="pt").input_ids
            vectores_salida = modelo.generate(vectores_entrada, max_length=150)
            resultado = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)
        elif eleccion == 3:
            contexto = request.form["contexto"]
            instrucciones = f"{tipo_tarea} {texto} context: {contexto}"
            vectores_entrada = convertir_vectores(instrucciones, return_tensors="pt").input_ids
            vectores_salida = modelo.generate(vectores_entrada, max_length=150)
            resultado = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)
        elif eleccion == 4:
            preguntas = generar_preguntas(texto)
            resultado = preguntas
        else:
            resultado = "Opción no válida."

        return render_template("index.html", resultado=resultado)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)