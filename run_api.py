"""
Standalone launcher for the FastAPI AI Text Detection service.
Runs uvicorn.run() directly so we get the full traceback on any startup error.
"""
import sys
import os
import traceback

# Make sure we're in the project root so relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(f"=== Launcher starting (CWD: {os.getcwd()}) ===", flush=True)

try:
    import uvicorn
    from api import app          # only fastapi/pydantic at this point
    print("Imports OK – starting uvicorn …", flush=True)
except BaseException:
    traceback.print_exc()
    sys.exit(1)

uvicorn.run(
    app,                          # pass the app object, not a string
    host="127.0.0.1",
    port=8000,
    reload=False,
    log_level="info",
)
