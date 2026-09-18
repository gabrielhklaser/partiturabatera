"""O coletor de resultados só pode apagar o que ele mesmo criou (docs/ERROS.md A13).

Foi a um passo de acontecer: `_store_pre_salvo` da demo aponta `path` para `samples/`, e o `_gc`
fazía `shutil.rmtree(dir)` de tudo que saísse do cache — o que, no 25º upload, levaria embora a
faixa de demonstração e o gabarito. Este teste encena exatamente isso com dois diretórios falsos:
um dentro de `out/uploads` (pode morrer) e um fora (não pode).
"""
import os, shutil, sys, time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from drumscribe import server                                   # noqa: E402

dentro = os.path.join(server.UPLOADS, "_gc_dentro")
fora = os.path.join(RAIZ, "out", "_gc_fora")
marca = os.path.join(fora, "demo_drums.wav")
falhas = []
try:
    os.makedirs(dentro, exist_ok=True)
    os.makedirs(fora, exist_ok=True)
    open(os.path.join(dentro, "arquivo.wav"), "wb").write(b"x")
    open(marca, "wb").write(b"nao me apague")
    # dois resultados "velhos", um com dir dentro de uploads e outro apontando para fora
    agora = time.time()
    server.RESULTS.clear()
    server.RESULTS["dentro"] = {"dir": dentro, "created": agora}
    server.RESULTS["fora"] = {"dir": fora, "created": agora - 10}
    server._gc(max_keep=0)
    if os.path.exists(dentro):
        falhas.append("o coletor não apagou o diretório de upload morto")
    if not os.path.exists(marca):
        falhas.append("O COLETOR APAGOU ARQUIVO FORA DE out/uploads — a guarda sumiu")
finally:
    shutil.rmtree(dentro, ignore_errors=True)
    shutil.rmtree(fora, ignore_errors=True)
    server.RESULTS.clear()

print("gc_guard_check: " + ("OK — só `out/uploads/<id>` é apagável, o resto sobrevive"
                            if not falhas else "FALHAS: " + "; ".join(falhas)))
sys.exit(1 if falhas else 0)
