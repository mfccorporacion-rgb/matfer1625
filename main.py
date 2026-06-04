from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import numpy as np
import cv2
import os
import json
import re
import pdfplumber

from google.cloud import vision
from google.oauth2 import service_account
from openai import OpenAI

app = FastAPI()

# =========================
# 🔐 GOOGLE CREDENTIALS
# =========================
cred_json = os.getenv("GOOGLE_CREDENTIALS")
if not cred_json:
    raise ValueError("Falta GOOGLE_CREDENTIALS")

cred_dict = json.loads(cred_json)
credentials = service_account.Credentials.from_service_account_info(cred_dict)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# =========================
# 🔐 OPENAI
# =========================
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 📋 CONSTANTES
# =========================
CLIENTE_NATURAL = "CERQUIN INFANTE LUIS FERNANDO"
RUC_NATURAL     = "10469879424"

CLIENTE_EMPRESA = "MFC CORPORACION E.I.R.L."
RUC_EMPRESA     = "20613822616"

# =========================
# 🧠 PREPROCESAMIENTO
# =========================
def mejorar_imagen_para_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

# =========================
# 📄 EXTRAER TEXTO PDF
# =========================
def extraer_texto_pdf(contents):
    texto = ""
    try:
        with pdfplumber.open(BytesIO(contents)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texto += t + "\n"
    except:
        pass
    return texto.strip()

# =========================
# 🔍 OCR IMAGEN
# =========================
def ocr_imagen(image_bytes):
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description

# =========================
# 🤖 IA (MEJORADO)
# =========================
def extraer_con_ia(texto):

    prompt = f"""
Extrae datos de una factura peruana.

REGLAS IMPORTANTES:

- Diferenciar EMISOR vs CLIENTE

CLIENTE POSIBLE:
- {CLIENTE_NATURAL}
- {CLIENTE_EMPRESA}

PRIORIDAD:
- Si aparece RUC {RUC_EMPRESA} → cliente = {CLIENTE_EMPRESA}
- Si aparece RUC {RUC_NATURAL} → cliente = {CLIENTE_NATURAL}

NUMERO DE FACTURA:
- Puede contener letras y números
- Ejemplo: F001-12345678, F74H-00003384, E001-00012345
- Puede tener espacios o guión
- Extraer exactamente como aparece
- NO inventar
- Si no existe devolver ""

CLASIFICACIÓN DE COMPRA:

Clasifica la compra según el tipo de producto o servicio.

Usa estas categorías:

COMIDA → restaurantes, fast food, pollerías
SUPERMERCADO → compras de víveres
COMBUSTIBLE → grifos, gasolina
TRANSPORTE → taxis, movilidad
SEGUROS → RIMAC, MAPFRE
SALUD → clínicas, hospitales
FARMACIA → boticas
TECNOLOGIA → equipos electrónicos
SERVICIOS → servicios generales
ALQUILER → renta de locales
MANTENIMIENTO → reparación
LIMPIEZA → productos limpieza
EDUCACION → cursos, universidades
BANCOS → comisiones bancarias
IMPUESTOS → SUNAT
ENTRETENIMIENTO → cine, ocio
VESTIMENTA → ropa
HERRAMIENTAS → ferretería
LOGISTICA → envíos, courier
OTROS → si no encaja

IMPORTANTE:
- Elegir SOLO UNA categoría
- NO inventar
- Basarse en el contenido de la factura

MONTOS:
- TOTAL: TOTAL, IMPORTE TOTAL, TOTAL A PAGAR
- IGV: IGV, I.G.V., IGV 18%

REGLAS NUMÉRICAS:
- SIEMPRE 2 decimales
- usar punto
- NO concatenar columnas

Devuelve SOLO JSON:

{{
"empresa": "",
"ruc": "",
"cliente": "",
"fecha": "",
"hora": "",
"total": "",
"igv": "",
"numero_factura": "",
"compra": ""
}}

Texto:
\"\"\"{texto}\"\"\"
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Devuelve SOLO JSON válido."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    contenido = response.choices[0].message.content.strip()

    if contenido.startswith("```"):
        contenido = re.sub(r"```(?:json)?", "", contenido).replace("```", "").strip()

    try:
        return json.loads(contenido)
    except:
        return {"error": "IA JSON inválido", "raw": contenido}

# =========================
# 🔧 VALIDACIÓN + CÁLCULO
# =========================
def validar(datos, texto):

    cliente = str(datos.get("cliente", "")).upper()

    if RUC_EMPRESA in texto:
        datos["cliente"] = CLIENTE_EMPRESA
        datos["ruc"] = RUC_EMPRESA

    elif RUC_NATURAL in texto:
        datos["cliente"] = CLIENTE_NATURAL
        datos["ruc"] = RUC_NATURAL

    if "MFC" in cliente:
        datos["cliente"] = CLIENTE_EMPRESA
        datos["ruc"] = RUC_EMPRESA

    if "INFANTE" in cliente:
        datos["cliente"] = CLIENTE_NATURAL
        datos["ruc"] = RUC_NATURAL

    def limpiar(valor):
        val = re.sub(r"[^\d.]", "", str(valor))
        if not val:
            return None
        num = float(val)
        if "." not in val:
            num = num / 100
        return round(num, 2)

    total = limpiar(datos.get("total"))
    igv = limpiar(datos.get("igv"))

    # 🔥 LÓGICA CONTABLE INTELIGENTE
    if total and not igv:
        subtotal = total / 1.18
        igv = round(total - subtotal, 2)

    elif igv and not total:
        subtotal = igv / 0.18
        total = round(subtotal * 1.18, 2)

    if total:
        datos["total"] = str(total)
    if igv:
        datos["igv"] = str(igv)

    return datos

# =========================
# 🚀 ENDPOINT
# =========================
@app.post("/procesar")
async def procesar(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        texto_detectado = ""

        if file.content_type == "application/pdf":

            texto_pdf = extraer_texto_pdf(contents)

            if len(texto_pdf) > 50:
                texto_detectado = texto_pdf
            else:
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    procesada = mejorar_imagen_para_ocr(img)
                    _, buffer = cv2.imencode(".jpg", procesada)
                    texto_detectado = ocr_imagen(buffer.tobytes())

        else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JSONResponse({"error": "Imagen inválida"}, status_code=400)

            procesada = mejorar_imagen_para_ocr(img)
            _, buffer = cv2.imencode(".jpg", procesada)
            texto_detectado = ocr_imagen(buffer.tobytes())

        if not texto_detectado:
            return {"error": "No se pudo extraer texto"}

        datos = extraer_con_ia(texto_detectado)
        datos = validar(datos, texto_detectado)

        return {"datos": datos}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)