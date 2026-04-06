# **Workshop Setup Guide**

Complete this **before** the workshop. Pulling Docker images on conference WiFi is not fun. Here are useful links:

* **github repo:** [github.com/nleroy917/biomedical-graphrag-workshop](http://github.com/nleroy917/biomedical-graphrag-workshop)  
* **Neo4J quickstart:** [https://neo4j.com/docs/getting-started/](https://neo4j.com/docs/getting-started/)  
* **Qdrant quickstart:** [https://qdrant.tech/documentation/quickstart/](https://qdrant.tech/documentation/quickstart/)  
* **Docker installation:** [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)  
* **Git installation:** [https://git-scm.com/install/](https://git-scm.com/install/)

---

## **Prerequisite software and API keys**

* Python 3.13  
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Linux) or Docker \+ WSL2 (Windows)  
* An OpenAI API key — grab one at [platform.openai.com](https://platform.openai.com/)  
* git

---

## **1\. Clone the Repo**

```shell
git clone github.com/nleroy917/biomedical-graphrag-workshop
cd biomedical-graphrag-workshop
```

---

## **2\. Python Environment**

```shell
python -m venv .venv
source .venv/bin/activate  # Windows (WSL): same command

# optional: use `uv`
pip install --upgrade pip
pip install -r requirements.txt
```

---

## **3\. Pull Docker Images**

Do this at home. Seriously.

```shell
docker pull qdrant/qdrant:latest
docker pull neo4j:5-community
```

You can verify they pulled correctly:

```shell
docker images | grep -E "qdrant|neo4j"
```

---

## **4\. Start the Containers**

```shell
docker compose up -d
```

This spins up:

* **Qdrant** on `http://localhost:6333` — UI available at `http://localhost:6333/dashboard`  
* **Neo4j** on `http://localhost:7474` — browser UI, credentials `neo4j / workshop123`

Verify both are running:

```shell
docker compose ps
```

Both should show `running`. If anything looks off:

```shell
docker compose logs qdrant
docker compose logs neo4j
```

---

## **5\. OpenAI API Key**

Copy the example env file and add your key:

```shell
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=sk-...
```

The notebooks load this file automatically. Alternatively, export it in your shell:

```shell
export OPENAI_API_KEY="sk-..."
```

---

## **6\. Sanity Check**

Run this python to confirm everything is talking to each other:

```py
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from openai import OpenAI

# Qdrant
q = QdrantClient(url='http://localhost:6333')
print('Qdrant:', q.get_collections())

# Neo4j
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'workshop123'))
driver.verify_connectivity()
print('Neo4j: connected')
driver.close()

# OpenAI
client = OpenAI()
print('OpenAI: key loaded')
print('All good.')

```

If you see `All good.` you are set.

---

## **Troubleshooting**

**Port already in use** Something else on your machine is occupying `6333` or `7474`. Either stop that process or change the ports in `docker-compose.yml`.

**Docker Desktop not running** Start Docker Desktop first, then re-run `docker compose up -d`.

**Neo4j asks you to change the password on first login** The `docker-compose.yml` sets the password to `workshop123` via `NEO4J_AUTH`. If you see a password prompt in the Neo4j browser UI, use `neo4j / workshop123`. If you started Neo4j without the compose file, the default is `neo4j / neo4j` and it will ask you to change it — update `.env` accordingly.

**OpenAI key not found** Make sure your `.env` file exists in the repo root with `OPENAI_API_KEY=sk-...`. The notebooks load it automatically. If running scripts outside notebooks, export it in your shell.

**WSL2 / Windows** All commands above work in a WSL2 terminal. Make sure Docker Desktop has WSL2 integration enabled under Settings → Resources → WSL Integration.