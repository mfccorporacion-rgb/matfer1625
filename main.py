from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from io import BytesIO
from datetime import datetime, timezone
import numpy as np
import cv2
import os
import json
import re
import secrets
import unicodedata
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
# 🔐 CLAVE DE ACCESO
# =========================
# Si API_KEY no esta definida en Render, el endpoint sigue abierto (modo
# compatible mientras se actualiza el Apps Script). Con API_KEY definida, el
# script debe enviarla en el encabezado X-API-KEY.
API_KEY = os.getenv("API_KEY", "")

def verificar_clave(x_api_key: str = Header(default="")):
    if API_KEY and not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="No autorizado")

# =========================
# 📋 CONSTANTES
# =========================
CLIENTE_NATURAL = "CERQUIN INFANTE LUIS FERNANDO"
RUC_NATURAL     = "10469879424"

CLIENTE_EMPRESA = "MFC CORPORACION E.I.R.L."
RUC_EMPRESA     = "20613822616"

CATEGORIAS = [
    "COMIDA", "SUPERMERCADO", "COMBUSTIBLE", "TRANSPORTE", "SEGUROS", "SALUD",
    "FARMACIA", "TECNOLOGIA", "SERVICIOS", "ALQUILER", "MANTENIMIENTO",
    "LIMPIEZA", "EDUCACION", "BANCOS", "IMPUESTOS", "ENTRETENIMIENTO",
    "VESTIMENTA", "HERRAMIENTAS", "LOGISTICA", "OTROS",
]

MAX_PAGINAS_PDF = 3

MESES = {
    "ENE": 1, "ENERO": 1, "JAN": 1, "JANUARY": 1,
    "FEB": 2, "FEBRERO": 2, "FEBRUARY": 2,
    "MAR": 3, "MARZO": 3, "MARCH": 3,
    "ABR": 4, "ABRIL": 4, "APR": 4, "APRIL": 4,
    "MAY": 5, "MAYO": 5,
    "JUN": 6, "JUNIO": 6, "JUNE": 6,
    "JUL": 7, "JULIO": 7, "JULY": 7,
    "AGO": 8, "AGOSTO": 8, "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SET": 9, "SETIEMBRE": 9, "SEPTIEMBRE": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTUBRE": 10, "OCTOBER": 10,
    "NOV": 11, "NOVIEMBRE": 11, "NOVEMBER": 11,
    "DIC": 12, "DICIEMBRE": 12, "DEC": 12, "DECEMBER": 12,
}

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
    except Exception:
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

def ocr_imagen_cv(img):
    procesada = mejorar_imagen_para_ocr(img)
    _, buffer = cv2.imencode(".jpg", procesada)
    return ocr_imagen(buffer.tobytes())

# =========================
# 🖨️ OCR PDF ESCANEADO
# =========================
def ocr_pdf_escaneado(contents):
    # PDF sin capa de texto: se renderiza cada pagina como imagen y se lee con Vision.
    texto = ""
    with pdfplumber.open(BytesIO(contents)) as pdf:
        for page in pdf.pages[:MAX_PAGINAS_PDF]:
            pil = page.to_image(resolution=200).original.convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            texto += ocr_imagen_cv(img) + "\n"
    return texto.strip()

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

FECHA:
- Formato de salida OBLIGATORIO: DD/MM/YYYY (día primero, como se usa en Perú).
- Convierte desde cualquier formato: DD/MM/YY, DD-MM-YYYY, DD.MM.YY, YYYY-MM-DD.
- También puede venir con el mes en letras, en español o inglés, abreviado o completo. Ejemplos: "04 NOV 2026" -> "04/11/2026", "4 de noviembre de 2026" -> "04/11/2026", "NOV 04, 2026" -> "04/11/2026", "04-SET-26" -> "04/09/2026"
- Si el año viene con 2 dígitos, conviértelo a 4 dígitos (26 → 2026). Ejemplo: "19/10/26" -> "19/10/2026"
- NUNCA interpretes la fecha como MM/DD/YYYY.
- Si no hay ninguna fecha impresa devuelve "" — no inventes una.

HORA:
- Busca la hora en TODO el documento, no solo cerca de la palabra "Hora". Es muy común que aparezca en alguno de estos lugares (boletas peruanas reales):
  1. Junto a la fecha, en la misma línea que el número de serie de la factura. Ejemplo: "F006-0004421 10/06/2026 22:27:24" -> hora = 22:27:24
  2. Con etiqueta explícita "HORA:" en el pie de página, junto a otros códigos de la transacción (terminal, lote, referencia). El año ahí suele venir en 2 dígitos. Ejemplo: "AP:072336 FECHA:23/06/26 HORA:17:07" -> hora = 17:07
  3. Junto a la fecha con puntos en vez de barras, sin ninguna etiqueta, casi siempre en una línea de códigos de terminal cerca del final. Ejemplo: "23.06.26 17:34" -> hora = 17:34
  4. Con sufijo AM/PM explícito, en cualquier variante de escritura ("PM", "pm", "p.m.", "p. m."). Ejemplo: "20/06/2026 01:54:51 PM" -> hora = 13:54:51 (convierte siempre a formato 24 horas)
- Siempre que encuentres una hora, conviértela a formato 24 horas HH:MM:SS (agrega ":00" de segundos si no vienen).
- Si después de revisar todo el documento genuinamente no hay ninguna hora impresa, devuelve "" — no inventes una.

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
        temperature=0,
        response_format={"type": "json_object"}
    )

    contenido = response.choices[0].message.content.strip()

    if contenido.startswith("```"):
        contenido = re.sub(r"```(?:json)?", "", contenido).replace("```", "").strip()

    try:
        return json.loads(contenido)
    except Exception:
        return {"error": "IA JSON inválido", "raw": contenido}

# =========================
# 🧹 NORMALIZADORES
# =========================
def normalizar_fecha(valor):
    # Devuelve DD/MM/YYYY o "" si la fecha no es valida o cae fuera del rango razonable.
    # Acepta numericas (19/10/26, 19-10-2026, 19.10.26, 2026-10-19) y con mes en
    # letras (04 NOV 2026, 4 de noviembre de 2026, NOV 04, 2026, 04-NOV-26).
    s = str(valor or "").strip()
    texto = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().upper()

    dia = mes = anio = None

    m = re.search(r"(\d{1,2})[\s\-/.]*(?:DE\s+)?([A-Z]{3,10})\.?[\s,\-/]*(?:DEL?\s+)?(\d{2,4})", texto)
    if m and m.group(2) in MESES:
        dia, mes, anio = int(m.group(1)), MESES[m.group(2)], int(m.group(3))
    else:
        m = re.search(r"([A-Z]{3,10})\.?\s*(\d{1,2})\s*,?\s*(\d{2,4})", texto)
        if m and m.group(1) in MESES:
            dia, mes, anio = int(m.group(2)), MESES[m.group(1)], int(m.group(3))

    if dia is None:
        m = re.search(r"(\d{1,4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,4})", s)
        if not m:
            return ""

        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(m.group(1)) == 4:
            anio, mes, dia = a, b, c
        else:
            dia, mes, anio = a, b, c
            if mes > 12 and dia <= 12:
                dia, mes = mes, dia

    if anio < 100:
        anio += 2000

    actual = datetime.now(timezone.utc).year
    if not (actual - 3 <= anio <= actual + 1):
        return ""

    try:
        datetime(anio, mes, dia)
    except ValueError:
        return ""

    return f"{dia:02d}/{mes:02d}/{anio}"

def normalizar_hora(valor):
    # Devuelve HH:MM:SS en 24 horas o "" si no hay una hora valida.
    s = str(valor or "").strip().lower()
    m = re.search(r"(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?", s)
    if not m:
        return ""

    h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)

    if re.search(r"p\s*\.?\s*m", s) and h < 12:
        h += 12
    elif re.search(r"a\s*\.?\s*m", s) and h == 12:
        h = 0

    if h > 23 or mi > 59 or se > 59:
        return ""

    return f"{h:02d}:{mi:02d}:{se:02d}"

_MESES_RE = "|".join(sorted(MESES, key=len, reverse=True))

def fecha_desde_texto(texto):
    # Respaldo cuando la IA no devuelve una fecha valida: se busca en el texto del
    # OCR, con los mismos formatos y en el mismo orden que FACTURAS OMAR. A
    # diferencia de OMAR, se prueban todas las coincidencias y se devuelve la
    # primera que sea una fecha valida y dentro del rango.
    t = unicodedata.normalize("NFD", str(texto or "")).encode("ascii", "ignore").decode().upper()
    patrones = [
        r"\b\d{2}[/.\-]\d{2}[/.\-]\d{4}\b",
        r"\b\d{4}[/.\-]\d{2}[/.\-]\d{2}\b",
        r"\b\d{2}[/.\-]\d{2}[/.\-]\d{2}\b",
        rf"\b\d{{1,2}}\s*(?:DE\s+)?(?:{_MESES_RE})\.?\s*(?:DE\s+)?\d{{4}}\b",
        rf"\b(?:{_MESES_RE})\.?\s+\d{{1,2}},?\s+\d{{4}}\b",
        rf"\b\d{{2}}(?:{_MESES_RE})\d{{2}}\b",
    ]
    for patron in patrones:
        for m in re.finditer(patron, t):
            fecha = normalizar_fecha(m.group(0))
            if fecha:
                return fecha
    return ""

def hora_desde_texto(texto):
    # Respaldo cuando la IA no devuelve hora. Solo toma horas con a. m./p. m.,
    # pegadas a una fecha, con etiqueta HORA o con segundos. No toma HH:MM sueltos
    # (podrian ser horarios de atencion) para no inventar una hora.
    hora = r"\d{1,2}[:.]\d{2}(?:[:.]\d{2})?"
    patrones = [
        rf"\b({hora}\s*[AaPp]\s*\.?\s*[Mm]\b\.?)",
        rf"\d{{1,4}}[/.\-]\d{{1,2}}[/.\-]\d{{2,4}}[ T]+({hora})",
        rf"HORA\s*:?\s*({hora})",
        r"\b(\d{1,2}:\d{2}:\d{2})\b",
    ]
    for patron in patrones:
        for m in re.finditer(patron, str(texto or ""), re.IGNORECASE):
            h = normalizar_hora(m.group(1))
            if h:
                return h
    return ""

def normalizar_compra(valor):
    s = unicodedata.normalize("NFD", str(valor or "")).encode("ascii", "ignore").decode()
    s = s.strip().upper()
    return s if s in CATEGORIAS else "OTROS"

def normalizar_numero_factura(valor):
    return re.sub(r"\s+", "", str(valor or "")).upper()

def limpiar_monto(valor, texto=""):
    val = re.sub(r"[^\d.,]", "", str(valor))
    if not val:
        return None

    pos_decimal = max(val.rfind(","), val.rfind("."))

    if pos_decimal == -1:
        # Sin separador decimal visible: solo se asume que el OCR perdio el
        # punto (ultimos 2 digitos = centavos) si el monto con decimales
        # aparece realmente en el texto. Si no, es un monto entero.
        n = int(val)
        con_decimales = rf"(?<!\d){n // 100}[.,]{n % 100:02d}(?!\d)"
        if re.search(con_decimales, texto):
            return round(n / 100, 2)
        return float(n)

    # El separador mas a la derecha es el decimal real; cualquier otro
    # separador antes de esa posicion es de miles y se descarta.
    entero = re.sub(r"[.,]", "", val[:pos_decimal]) or "0"
    decimal = re.sub(r"[.,]", "", val[pos_decimal + 1:])
    if len(decimal) == 0:
        decimal = "00"
    elif len(decimal) == 1:
        decimal = decimal + "0"
    elif len(decimal) > 2:
        entero = entero + decimal
        decimal = "00"

    return round(float(entero + "." + decimal), 2)

# =========================
# 🔧 VALIDACIÓN + CÁLCULO
# =========================
def validar(datos, texto):

    advertencias = []
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

    fecha_original = str(datos.get("fecha") or "").strip()
    datos["fecha"] = normalizar_fecha(fecha_original)
    if not datos["fecha"]:
        datos["fecha"] = fecha_desde_texto(texto)
        if datos["fecha"]:
            advertencias.append(f"Fecha tomada del texto (la IA devolvió: {fecha_original or 'vacío'})")
        elif fecha_original:
            advertencias.append(f"Fecha no válida o fuera de rango: {fecha_original}")

    hora_original = str(datos.get("hora") or "").strip()
    datos["hora"] = normalizar_hora(hora_original)
    if not datos["hora"]:
        datos["hora"] = hora_desde_texto(texto)
        if datos["hora"]:
            advertencias.append(f"Hora tomada del texto (la IA devolvió: {hora_original or 'vacío'})")
        elif hora_original:
            advertencias.append(f"Hora no válida: {hora_original}")

    datos["compra"] = normalizar_compra(datos.get("compra"))
    datos["numero_factura"] = normalizar_numero_factura(datos.get("numero_factura"))

    total = limpiar_monto(datos.get("total"), texto)
    igv = limpiar_monto(datos.get("igv"), texto)
    sin_igv = re.search(r"EXONERAD|INAFECT", texto.upper())

    # 🔥 LÓGICA CONTABLE INTELIGENTE
    if total and igv is None:
        if sin_igv:
            igv = 0.0
        else:
            igv = round(total - total / 1.18, 2)
            advertencias.append("IGV calculado a partir del total (no se leyó en el documento)")

    elif igv and not total:
        total = round(igv / 0.18 * 1.18, 2)
        advertencias.append("Total calculado a partir del IGV (no se leyó en el documento)")

    if total and igv and not sin_igv:
        esperado = total * 0.18 / 1.18
        if igv > total or abs(igv - esperado) > max(0.10, total * 0.02):
            advertencias.append("El IGV no cuadra con el total")

    if total is not None:
        datos["total"] = f"{total:.2f}"
    if igv is not None:
        datos["igv"] = f"{igv:.2f}"

    for campo, etiqueta in (
        ("fecha", "fecha"),
        ("hora", "hora"),
        ("empresa", "empresa"),
        ("total", "total"),
        ("numero_factura", "número de factura"),
    ):
        if not str(datos.get(campo) or "").strip():
            advertencias.append(f"Falta {etiqueta}")

    datos["advertencias"] = advertencias
    return datos

# =========================
# 🚀 ENDPOINT
# =========================
@app.get("/")
def estado():
    return {"estado": "activo"}

@app.post("/procesar", dependencies=[Depends(verificar_clave)])
async def procesar(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        texto_detectado = ""

        es_pdf = file.content_type == "application/pdf" or contents[:4] == b"%PDF"

        if es_pdf:

            texto_pdf = extraer_texto_pdf(contents)

            if len(texto_pdf) > 50:
                texto_detectado = texto_pdf
            else:
                texto_detectado = ocr_pdf_escaneado(contents)

        else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JSONResponse({"error": "Imagen inválida"}, status_code=400)

            texto_detectado = ocr_imagen_cv(img)

        if not texto_detectado:
            return {"error": "No se pudo extraer texto"}

        datos = extraer_con_ia(texto_detectado)

        if "error" in datos:
            return JSONResponse(datos, status_code=502)

        datos = validar(datos, texto_detectado)

        return {"datos": datos}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
