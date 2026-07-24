"""
App: Imagen -> Modelo 3D (GLB con textura + STL geometría)
Usa el espacio gratuito de Hugging Face "stabilityai/TripoSR" (mismo tipo de
modelo generativo que usa Tripo3D) a través de gradio_client, y convierte
el resultado a STL con trimesh.

STL no soporta color/textura (es un formato de solo geometría), por eso
la app entrega DOS archivos:
  - modelo.glb  -> con textura, para visualizar / usar en otros programas
  - modelo.stl  -> solo geometría, listo para impresión 3D
"""

import os
import uuid
import shutil
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from gradio_client import Client, handle_file
import trimesh

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_SPACE = "stabilityai/TripoSR"

app = FastAPI(title="Imagen a STL 3D")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.post("/convert")
async def convert_image_to_3d(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        client = Client(HF_SPACE)

        preprocessed = client.predict(
            handle_file(input_path),
            True,
            0.85,
            api_name="/preprocess",
        )

        result = client.predict(
            preprocessed,
            64,
            True,
            api_name="/generate",
        )

        generated_path = result if isinstance(result, str) else result[0]

        glb_out = os.path.join(job_dir, "modelo.glb")
        stl_out = os.path.join(job_dir, "modelo.stl")

        mesh = trimesh.load(generated_path, force="mesh")

        mesh.export(glb_out)
        mesh.export(stl_out)

        return JSONResponse({
            "job_id": job_id,
            "glb_url": f"/download/{job_id}/glb",
            "stl_url": f"/download/{job_id}/stl",
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generando el modelo 3D: {e}")


@app.get("/download/{job_id}/{fmt}")
def download(job_id: str, fmt: str):
    filename = "modelo.glb" if fmt == "glb" else "modelo.stl"
    path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(path, filename=filename)
