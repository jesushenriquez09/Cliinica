import spacy

nlp = spacy.load("es_core_news_md")

texto = """ 
Tengo dolor al orinar y molestia en la vejiga desde hace dos días.
""" 
def entidades_nlp(texto:str):
    doc = nlp(texto)

    print("entidades nombradas:")

    for en in doc.ents:
        print(f"texto:{en.text}, tipo:{en.label_}")

    for label in set(ent.label_ for ent in doc.ents):
        print(f"{label}:{spacy.explain(label)}")
        return {
            "entidades": [{ "texto": ent.text, "tipo": ent.label_ } for ent in doc.ents]
        }
        
''' resultado = entidades_nlp(texto)
print(resultado) '''                                                                
    