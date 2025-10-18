import spacy
import nltk
from datetime import timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Body
from transformers import pipeline
from textblob import TextBlob
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from config.db import SessionLocal, get_db
from modelo.oauth import get_current_user
from models.db_p import Users, Historial, Diagnostico, Citas, Entidad
from sqlalchemy.orm import Session
from modelo import m_pro
from nltk.probability import FreqDist 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json

class ProcesadorTextoMedico:
    def __init__(self):
        # Inicializar modelos de forma lazy
        self.nlp = None
        self.sentiment_analyzer = None
        self.translator = None
        
        # Asegurar recursos de NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('spanish'))
        except:
            print("Error al cargar stopwords en español")
            self.stop_words = set()

    def get_nlp(self):
        """Inicialización lazy de spaCy con patrones personalizados"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("es_core_news_md")
                
                # Definir patrones para términos médicos
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                patrones = [
                    # Síntomas comunes
                    {"label": "SINTOMA", "pattern": "dolor"},
                    {"label": "SINTOMA", "pattern": "dolor de cabeza"},
                    {"label": "SINTOMA", "pattern": "fiebre"},
                    {"label": "SINTOMA", "pattern": "malestar"},
                    {"label": "SINTOMA", "pattern": "náuseas"},
                    {"label": "SINTOMA", "pattern": "vómitos"},
                    {"label": "SINTOMA", "pattern": "mareos"},
                    {"label": "SINTOMA", "pattern": "fatiga"},
                    {"label": "SINTOMA", "pattern": "ardor"},
                    {"label": "SINTOMA", "pattern": "picazón"},
                    
                    # Partes del cuerpo
                    {"label": "CUERPO", "pattern": "cabeza"},
                    {"label": "CUERPO", "pattern": "estómago"},
                    {"label": "CUERPO", "pattern": "vejiga"},
                    {"label": "CUERPO", "pattern": "riñón"},
                    {"label": "CUERPO", "pattern": "riñones"},
                    {"label": "CUERPO", "pattern": "garganta"},
                    {"label": "CUERPO", "pattern": "espalda"},
                    {"label": "CUERPO", "pattern": "pecho"},
                    {"label": "CUERPO", "pattern": "abdomen"},
                    {"label": "CUERPO", "pattern": "pierna"},
                    {"label": "CUERPO", "pattern": "brazo"},
                    {"label": "CUERPO", "pattern": "uretra"},
                    
                    # Cualificadores
                    {"label": "INTENSIDAD", "pattern": "leve"},
                    {"label": "INTENSIDAD", "pattern": "moderado"},
                    {"label": "INTENSIDAD", "pattern": "intenso"},
                    {"label": "INTENSIDAD", "pattern": "severo"},
                    
                    # Tiempo
                    {"label": "TIEMPO", "pattern": "días"},
                    {"label": "TIEMPO", "pattern": "semanas"},
                    {"label": "TIEMPO", "pattern": "meses"},
                    {"label": "TIEMPO", "pattern": "horas"},
                    
                    # Medicamentos comunes
                    {"label": "MEDICAMENTO", "pattern": "paracetamol"},
                    {"label": "MEDICAMENTO", "pattern": "ibuprofeno"},
                    {"label": "MEDICAMENTO", "pattern": "antibiótico"},
                    {"label": "MEDICAMENTO", "pattern": "aspirina"}
                ]
                
                ruler.add_patterns(patrones)
                print("Patrones de entidades médicas añadidos correctamente")
            except Exception as e:
                print(f"Error cargando spaCy: {str(e)}")
        return self.nlp

    def get_sentiment_analyzer(self):
        """Inicialización lazy del analizador de sentimientos"""
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="finiteautomata/beto-sentiment-analysis"
                )
            except Exception as e:
                print(f"Error cargando analizador de sentimientos: {str(e)}")
        return self.sentiment_analyzer

    def get_translator(self):
        """Inicialización lazy del traductor"""
        if self.translator is None:
            try:
                self.translator = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-es-en"
                )
            except Exception as e:
                print(f"Error cargando traductor: {str(e)}")
        return self.translator

# Crear una instancia global del procesador
procesador = ProcesadorTextoMedico()

# Configurar el router con tags para mejor documentación
nlp_route = APIRouter(
    prefix="/nlp",
    tags=["NLP"],
    responses={404: {"description": "Not found"}},
)

@nlp_route.post("/", response_model=m_pro.vhBase,
                summary="Procesar texto con NLP",
                description="Analiza un texto en español utilizando técnicas de NLP para extraer información relevante")
async def process_text(
    texto: m_pro.nlp_create = Body(..., example={"texto_original": "Este es un ejemplo de texto para analizar", "cita_id": None, "diagnostico": None}),
    cita_id: Optional[int] = Body(None),
    diagnostico: Optional[str | int] = Body(None),
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Procesa un texto utilizando técnicas avanzadas de NLP:
    
    - Genera un resumen
    - Detecta entidades nombradas
    - Extrae palabras clave
    - Analiza el sentimiento
    - Traduce si es necesario
    
    Returns:
        m_pro.vhBase: Resultado del análisis
    """
    try:
        texto_original = texto.texto_original
        
        # 1. Obtener el modelo spaCy
        nlp = procesador.get_nlp()
        if nlp is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo spaCy no disponible. Instálalo con: python -m spacy download es_core_news_md"
            )
        
        # 2. Procesar el texto con spaCy
        doc = nlp(texto_original)
        
        # 3. Generar resumen
        resumen = str(list(doc.sents)[0]) if len(list(doc.sents)) > 0 else texto_original[:100]
        
        # 4. Detectar idioma y traducir si es necesario
        try:
            blob = TextBlob(texto_original)
            # Usar translate directamente, TextBlob intentará detectar el idioma automáticamente
            try:
                traduccion = str(blob.translate(to='es'))
                # Si la traducción es igual al texto original, significa que ya estaba en español
                if traduccion == texto_original:
                    print("El texto ya está en español")
            except:
                print("No se pudo traducir, el texto probablemente ya está en español")
                traduccion = texto_original
        except Exception as e:
            print(f"Error en traducción: {str(e)}")
            traduccion = texto_original
        
        # 5. Extraer entidades del texto original
        entidades_list = []
        entidades_map = {}
        
        # Diccionario de términos médicos y sus categorías
        terminos_medicos = {
            # Síntomas
            "dolor": "SINTOMA",
            "ardor": "SINTOMA",
            "picazón": "SINTOMA",
            "malestar": "SINTOMA",
            "molestia": "SINTOMA",
            "inflamación": "SINTOMA",
            "hinchazón": "SINTOMA",
            "fiebre": "SINTOMA",
            
            # Partes del cuerpo
            "vejiga": "PARTE_CUERPO",
            "riñón": "PARTE_CUERPO",
            "riñones": "PARTE_CUERPO",
            "uretra": "PARTE_CUERPO",
            "cabeza": "PARTE_CUERPO",
            "espalda": "PARTE_CUERPO",
            "abdomen": "PARTE_CUERPO",
            "estómago": "PARTE_CUERPO",
            
            # Tiempo
            "días": "TIEMPO",
            "semanas": "TIEMPO",
            "meses": "TIEMPO",
            "horas": "TIEMPO",
            
            # Intensidad
            "leve": "INTENSIDAD",
            "moderado": "INTENSIDAD",
            "intenso": "INTENSIDAD",
            "fuerte": "INTENSIDAD",
            "severo": "INTENSIDAD"
        }
        
        # Tokenizar el texto
        palabras = word_tokenize(texto_original.lower())
        
        # Buscar términos médicos en el texto
        i = 0
        while i < len(palabras):
            # Intentar encontrar frases de dos o tres palabras
            for longitud in [3, 2, 1]:
                if i + longitud <= len(palabras):
                    frase = " ".join(palabras[i:i + longitud])
                    if frase in terminos_medicos:
                        categoria = terminos_medicos[frase]
                        entrada = {
                            "label": categoria,
                            "texto": frase
                        }
                        if entrada not in entidades_list:
                            entidades_list.append(entrada)
                            if categoria not in entidades_map:
                                entidades_map[categoria] = []
                            if frase not in entidades_map[categoria]:
                                entidades_map[categoria].append(frase)
                        i += longitud - 1
                        break
            i += 1
        
        # Buscar patrones específicos (dolor/molestia en [parte del cuerpo])
        for i, palabra in enumerate(palabras):
            if palabra in ["dolor", "molestia"] and i + 2 < len(palabras):
                if palabras[i + 1] in ["en", "de"]:
                    parte_cuerpo = palabras[i + 2]
                    if parte_cuerpo in terminos_medicos and terminos_medicos[parte_cuerpo] == "PARTE_CUERPO":
                        # Agregar el síntoma
                        entrada_sintoma = {
                            "label": "SINTOMA",
                            "texto": palabra
                        }
                        if entrada_sintoma not in entidades_list:
                            entidades_list.append(entrada_sintoma)
                        
                        # Agregar la parte del cuerpo
                        entrada_cuerpo = {
                            "label": "PARTE_CUERPO",
                            "texto": parte_cuerpo
                        }
                        if entrada_cuerpo not in entidades_list:
                            entidades_list.append(entrada_cuerpo)
        
        # Debug: imprimir entidades encontradas
        print("Entidades encontradas:")
        for ent in entidades_list:
            print(f"- {ent['label']}: {ent['texto']}")
        
        # Convertir a JSON para almacenamiento
        entidades = json.dumps({
            "entidades": entidades_list,
            "agrupadas": entidades_map
        }, ensure_ascii=False, indent=2)
        
        # 6. Extraer palabras clave
        palabras_claves = {}
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ', 'VERB']:
                if token.lemma_ not in palabras_claves:
                    palabras_claves[token.lemma_] = 1
                else:
                    palabras_claves[token.lemma_] += 1
        palabras_claves = str(dict(sorted(palabras_claves.items(), key=lambda x: x[1], reverse=True)[:5]))
        
        # 7. Análisis de sentimiento
        try:
            analyzer = procesador.get_sentiment_analyzer()
            if analyzer:
                sentiment_result = analyzer(texto_original)[0]
                sentimiento = "positivo" if sentiment_result['label'] == 'POS' else "negativo" if sentiment_result['label'] == 'NEG' else "neutral"
            else:
                sentimiento = "neutral"
        except Exception as e:
            print(f"Error en análisis de sentimiento: {str(e)}")
            sentimiento = "neutral"
            
        # 8. Diagnóstico automático
        # Obtener palabras del texto del paciente y filtrar palabras no significativas
        palabras_no_significativas = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'de', 
            'del', 'en', 'por', 'con', 'al', 'a', 'que', 'es', 'son', 'ser',
            'para', 'como', 'se', 'ha', 'he', 'hay', 'hace', 'desde', 'este', 'esta',
            'estos', 'estas'
        }
        
        # Tokenizar y procesar el texto del paciente
        texto_procesado = texto_original.lower()
        # Reemplazar algunos términos comunes
        texto_procesado = texto_procesado.replace("tengo", "")
        
        tokens = word_tokenize(texto_procesado)
        palabras_texto = {word for word in tokens 
                         if word.isalnum() 
                         and word not in palabras_no_significativas}
        
        print(f"Palabras del texto: {palabras_texto}")  # Debug
        
        # Obtener todos los diagnósticos
        todos_diagnosticos = db.query(Diagnostico).all()
        
        # Variables para encontrar el mejor match
        mejor_match = None
        max_coincidencias = 0
        max_porcentaje = 0
        umbral_minimo = 1  # Reducido a 1 palabra coincidente
        umbral_porcentaje = 0.2  # Reducido a 20% de coincidencia
        
        for diagnostico in todos_diagnosticos:
            # Procesar el texto del diagnóstico
            texto_diagnostico = diagnostico.diagnostico.lower()
            palabras_diagnostico = {word for word in word_tokenize(texto_diagnostico)
                                  if word.isalnum() 
                                  and word not in palabras_no_significativas}
            
            # Calcular coincidencias
            coincidencias = palabras_texto & palabras_diagnostico  # Intersección
            num_coincidencias = len(coincidencias)
            porcentaje = num_coincidencias / len(palabras_diagnostico) if palabras_diagnostico else 0
            
            print(f"Diagnóstico: {diagnostico.diagnostico}")  # Debug
            print(f"Palabras diagnóstico: {palabras_diagnostico}")  # Debug
            print(f"Coincidencias: {coincidencias}, Porcentaje: {porcentaje}")  # Debug
            
            # Actualizar si encontramos mejores coincidencias
            if (num_coincidencias >= umbral_minimo and 
                porcentaje >= umbral_porcentaje and 
                (num_coincidencias > max_coincidencias or 
                 (num_coincidencias == max_coincidencias and porcentaje > max_porcentaje))):
                max_coincidencias = num_coincidencias
                max_porcentaje = porcentaje
                mejor_match = diagnostico
        
        # Asignar diagnóstico
        if mejor_match and max_coincidencias > 0:
            diagnostico_id = mejor_match.id
            print(f"Diagnóstico asignado: {mejor_match.diagnostico} (ID: {diagnostico_id})")
        else:
            # Buscar diagnóstico "sin diagnostico"
            diag_sin = db.query(Diagnostico).filter(Diagnostico.diagnostico.ilike('%sin diagnostico%')).first()
            if diag_sin:
                diagnostico_id = diag_sin.id
                print(f"Diagnóstico por defecto: {diag_sin.diagnostico} (ID: {diagnostico_id})")
            else:
                diagnostico_id = None
        
        # Validar cita si se proporciona
        if cita_id is not None:
            cita_obj = db.query(Citas).filter(Citas.id == cita_id).first()
            if not cita_obj:
                raise HTTPException(status_code=400, detail="Cita no encontrada")
        # 9. Crear objeto de respuesta y guardar en la base de datos
        resultado = m_pro.vhBase(
            texto_original=texto_original,
            resumen=resumen,
            traduccion=traduccion,
            entidades=entidades,
            palabras_claves=palabras_claves,
            sentimiento=sentimiento,
            diagnosticos_id=diagnostico_id,
            cita_id=cita_id,
            user_id=current_user.id
        )

        # Crear registro en el historial
        db_item = Historial(
            user_id=current_user.id,
            texto_original=resultado.texto_original,
            resumen=resultado.resumen,
            traduccion=resultado.traduccion,
            entidades=entidades,
            palabras_claves=resultado.palabras_claves,
            sentimiento=resultado.sentimiento,
            diagnosticos_id=diagnostico_id,
            cita_id=cita_id
        )

        # Guardar historial y obtener ID
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

        # Guardar entidades relacionadas
        entidades_data = json.loads(entidades)
        for ent in entidades_data["entidades"]:
            if ent["texto"].strip():  # Verificar que el texto no esté vacío
                nueva_ent = Entidad(
                    historial_id=db_item.id,
                    label=ent["label"],
                    texto=ent["texto"]
                )
                db.add(nueva_ent)
        
        try:
            db.commit()
            print(f"Se guardaron {len(entidades_data['entidades'])} entidades")
        except Exception as e:
            print(f"Error al guardar entidades: {str(e)}")
            db.rollback()

        # Preparar respuesta final
        resultado.user_id = db_item.user_id
        
        # Si no hay diagnóstico, indicar que está sano
        if resultado.diagnosticos_id is None:
            return {
                **resultado.model_dump(),
                "mensaje": "No se encontraron síntomas significativos. El paciente parece estar sano."
            }
        
        return resultado
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el texto: {str(e)}"
        )
    
    