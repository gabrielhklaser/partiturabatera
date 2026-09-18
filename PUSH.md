# Primeiro push

O histórico deste projeto foi montado em `main`, com um único commit inicial, aqui no ambiente
onde ele foi escrito. Faltou só autenticar o `git push` — este ambiente não tem chave SSH nem
token, e o GitHub não aceita escrita anônima.

Duas formas de levar adiante, ambas a partir de um clone/checkout destes arquivos:

```bash
# 1) direto desta pasta (precisa de um token com escopo Contents:write no repositório)
git remote add origin https://github.com/gabrielhklaser/partiturabatera.git
git push -u origin main

# 2) levando o histórico pronto pelo bundle (partiturabatera.bundle)
git clone partiturabatera.bundle partiturabatera
cd partiturabatera
git remote add origin https://github.com/gabrielhklaser/partiturabatera.git
git push -u origin main
```

Com token de acesso pessoal, a URL autenticada é
`https://<TOKEN>@github.com/gabrielhklaser/partiturabatera.git` — de preferência um
*fine-grained token*, válido por 1 dia, restrito a este repositório e a `Contents: Read and write`.
Depois do push, revogue-o.

## O que está de fora, e por escolha

* `out/` — logs, `out/uploads/` (as faixas que sobem pela interface e os PDFs gerados) e o cache
  de envelope de `out/wave/`. É tudo derivado e contém áudio de terceiros; o servidor recria o
  diretório no boot (`server.py` chama `os.makedirs`), então um clone limpo funciona sem ele.
* `samples/exemplo.mp3` (17,7 MB) — a faixa real usada para reproduzir os defeitos de upload e de
  silêncio inicial. É gravação de pessoa, não ativo do projeto, e o repositório é público.
  Para incluir mesmo assim: `git add -f samples/exemplo.mp3`.
* `__pycache__/` e `*.log`.

## Antes do push, vale conferir

```bash
python3 -m pip install -r requirements.txt
python3 tests/selfcheck.py && python3 tests/doublecheck.py && python3 tests/long_check.py
```

*Pode apagar este arquivo depois do primeiro push.*
