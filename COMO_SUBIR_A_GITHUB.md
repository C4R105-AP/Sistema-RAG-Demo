# 📤 Guía: Cómo Subir el Proyecto a GitHub

## 📋 Antes de Empezar

### ✅ Checklist

- [ ] Tienes cuenta en GitHub ([crear aquí](https://github.com/signup))
- [ ] Git está instalado ([descargar aquí](https://git-scm.com/downloads))
- [ ] Has comprimido `Sistema_RAG_Portable/` en `.zip`
- [ ] Has eliminado tu API key del código (el `.gitignore` te protege)

---

## 🚀 Paso 1: Crear Repositorio en GitHub

1. Ve a https://github.com/new
2. Configura:
   - **Nombre:** `Sistema-RAG-Demo` (o el que prefieras)
   - **Descripción:** `Sistema RAG con FAISS y LLMs - Ejecutable Windows`
   - **Visibilidad:** `Public` ✅ (para compartir)
   - **NO marques** "Add a README" (ya lo tienes)
3. Click en **Create repository**

---

## 💻 Paso 2: Inicializar Git Localmente

Abre PowerShell en tu carpeta del proyecto:

```powershell
cd C:\SCRIPTS\_RAG_Ejemplo
```

### Inicializar repositorio Git

```powershell
git init
git add .
git commit -m "Initial commit: Sistema RAG con ejecutable"
```

### Conectar con GitHub

**Reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub:**

```powershell
git remote add origin https://github.com/TU_USUARIO/Sistema-RAG-Demo.git
git branch -M main
git push -u origin main
```

---

## 📦 Paso 3: Crear Release con Ejecutable

### 3.1 Comprimir el Ejecutable

```powershell
# En PowerShell
Compress-Archive -Path Sistema_RAG_Portable -DestinationPath Sistema_RAG_Portable.zip
```

### 3.2 Subir Release en GitHub

1. Ve a tu repositorio en GitHub
2. Click en **Releases** (barra lateral derecha)
3. Click en **Create a new release**
4. Configura:
   - **Tag:** `v1.0.0`
   - **Title:** `Sistema RAG v1.0.0 - Windows`
   - **Description:**
   
```markdown
## 🚀 Primera versión estable

### ✨ Características
- ✅ Búsqueda semántica con FAISS
- ✅ Soporte OpenAI, Anthropic, Hugging Face
- ✅ Interfaz web moderna
- ✅ No requiere instalación de Python

### 📥 Instalación
1. Descarga `Sistema_RAG_Portable.zip`
2. Descomprime
3. Ejecuta `Sistema_RAG.exe`

### 🔧 Requisitos
- Windows 10/11 (64-bit)
- 8GB RAM recomendado

### 📚 Documentación
Ver [README.md](https://github.com/TU_USUARIO/Sistema-RAG-Demo#readme)
```

5. **Arrastra** `Sistema_RAG_Portable.zip` a la sección "Attach binaries"
6. Click en **Publish release**

---

## 🎯 Paso 4: Verificar Todo Funciona

### Tu repositorio ahora tiene:

```
https://github.com/TU_USUARIO/Sistema-RAG-Demo
├── README.md                 ← Descripción y guía
├── .gitignore                ← Protege credenciales
├── api_rag.py               ← Código fuente (SIN API keys)
├── interfaz_web.html        ← Interfaz web
├── requirements.txt         ← Dependencias
└── [Release] v1.0.0         ← Ejecutable descargable
    └── Sistema_RAG_Portable.zip
```

### Compartir con otros

Envía este link:
```
https://github.com/TU_USUARIO/Sistema-RAG-Demo/releases/latest
```

---

## 🔐 IMPORTANTE: Seguridad

### ❌ NUNCA subas esto a GitHub:

- ❌ API Keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- ❌ Archivos `.env`
- ❌ Tokens personales
- ❌ Contraseñas

### ✅ El `.gitignore` protege automáticamente:

- ✅ Variables de entorno (`.env`)
- ✅ Configuración local (`config.bat`)
- ✅ Virtual environment (`venv/`)
- ✅ Archivos compilados (`__pycache__/`)
- ✅ Builds de PyInstaller (`build/`, `dist/`)

### 🔍 Verificar antes de subir

```powershell
# Ver qué archivos se subirán
git status

# Ver contenido que se subirá
git diff --cached

# Si ves algo sensible, NO subas y agrégalo a .gitignore
```

---

## 🛠️ Comandos Útiles Git

### Ver estado actual
```powershell
git status
```

### Agregar cambios
```powershell
# Todos los archivos
git add .

# Solo un archivo
git add api_rag.py
```

### Hacer commit
```powershell
git commit -m "Descripción de cambios"
```

### Subir a GitHub
```powershell
git push
```

### Actualizar desde GitHub
```powershell
git pull
```

---

## 📝 Actualizar tu Proyecto

### Cuando hagas cambios:

```powershell
# 1. Agregar archivos modificados
git add .

# 2. Commit con mensaje descriptivo
git commit -m "Mejora: respuestas más rápidas con cache"

# 3. Subir a GitHub
git push

# 4. Si creaste nuevo ejecutable, crear nuevo Release
#    → GitHub > Releases > New release > v1.1.0
```

---

## 🎨 Personalizar tu README

Edita `README.md` y cambia:

1. **TU_USUARIO** → Tu username de GitHub
2. **Tu Nombre** → Tu nombre real
3. **Capturas de pantalla:** Agrega imágenes de la interfaz
4. **Badges:** Agrega badges de status

### Ejemplo de badges:

```markdown
![GitHub release](https://img.shields.io/github/v/release/TU_USUARIO/Sistema-RAG-Demo)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-blue)
```

---

## 📸 Agregar Screenshots (Opcional)

### 1. Toma capturas de tu interfaz

```powershell
# Crea carpeta para imágenes
mkdir docs/screenshots
```

### 2. Agrega las imágenes al repo

```powershell
git add docs/screenshots/*.png
git commit -m "Docs: agregar screenshots"
git push
```

### 3. Úsalas en el README

```markdown
## 📸 Capturas de Pantalla

![Interfaz principal](docs/screenshots/interfaz.png)
![Respuesta ejemplo](docs/screenshots/respuesta.png)
```

---

## 🌟 Promocionar tu Proyecto

### En GitHub:

1. **Topics:** Agrega etiquetas (Settings > Topics)
   - `rag`, `langchain`, `faiss`, `llm`, `python`, `fastapi`

2. **About:** Agrega descripción corta

3. **Website:** Si tienes demo online, agrégalo

### En LinkedIn/Twitter:

```
🚀 Acabo de publicar mi Sistema RAG con IA!

✨ Características:
• Búsqueda semántica en documentos
• Integración con GPT/Claude
• Ejecutable standalone (no requiere Python)

🔗 [Link a tu repo]

#AI #MachineLearning #RAG #Python #OpenSource
```

---

## ❓ Solución de Problemas

### Error: "remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/TU_USUARIO/Sistema-RAG-Demo.git
```

### Error: "Permission denied"
```powershell
# Necesitas autenticarte
# Opción 1: SSH keys (recomendado)
# Opción 2: Personal Access Token
# Ver: https://docs.github.com/en/authentication
```

### Error: "large files"
```powershell
# GitHub tiene límite de 100MB por archivo
# Si tu .exe es muy grande, usa Git LFS:
git lfs install
git lfs track "*.exe"
git add .gitattributes
```

---

## ✅ ¡Listo!

Tu proyecto ahora está en GitHub y otros pueden:
- ✅ Ver el código fuente
- ✅ Descargar el ejecutable
- ✅ Reportar issues
- ✅ Hacer fork y contribuir
- ✅ Dar estrella ⭐

**Link de tu repo:**
```
https://github.com/TU_USUARIO/Sistema-RAG-Demo
```

---

## 🎓 Recursos Adicionales

- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [GitHub Desktop](https://desktop.github.com/) - GUI para Git
- [Markdown Guide](https://www.markdownguide.org/) - Para mejorar tu README
- [Shields.io](https://shields.io/) - Generador de badges

---

**¿Dudas?** Abre un issue en tu repo y la comunidad te ayudará 💪

