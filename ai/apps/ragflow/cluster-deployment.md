---
title: "From Elasticsearch Queries to Kubernetes Cluster Deployment: A RAGFlow Journey"
date: 2026-03-18
tags: [elasticsearch, kubernetes, docker, helm, ragflow, deployment]
---

# From Elasticsearch Queries to Kubernetes Cluster Deployment: A RAGFlow Journey

When deploying a RAG (Retrieval-Augmented Generation) system like [RAGFlow](https://github.com/infiniflow/ragflow) in production, you inevitably encounter two major themes: **getting search right** and **getting infrastructure right**. This post walks through both — starting with the nuances of Elasticsearch queries in RAGFlow, then progressing to Kubernetes cluster deployment.

---

## Part 1: Understanding Elasticsearch in RAGFlow

### The Document Structure

A typical RAGFlow document stored in Elasticsearch looks like this:

```
doc_id, kb_id, docnm_kwd, authors_tks, title_tks, title_sm_tks,
authors_sm_tks, page_num_int, position_int, top_int,
content_with_weight, content_ltks, content_sm_ltks,
create_time, create_timestamp_flt, img_id, q_2560_vec
```

Notice what's *not* here: there is no `query` field. This means you cannot use `query_string` to search across all fields. Instead, you need to target specific indexed fields explicitly.

### Writing a Search Function

A reasonable first attempt uses `multi_match`:

```python
def search(es: Elasticsearch, index: str, query: str) -> None:
    if not query:
        return

    searchable_fields = [
        "content_ltks",
        "content_sm_ltks",
        "title_tks",
        "title_sm_tks",
        "authors_tks",
        "authors_sm_tks",
    ]

    res = es.search(
        index=index,
        query={
            "multi_match": {
                "query": query,
                "fields": searchable_fields,
                "type": "best_fields",
            }
        },
    )
    data = res.body if hasattr(res, "body") else res
    print_json(f"Search results for query: {query}", data)
```

**Key pitfall:** Some fields like `content_with_weight` have `"index": false` in the mapping. Including them in a `multi_match` query causes a `400 BadRequestError`. Always check field mappings before querying.

**Tip for keyword fields:** Fields ending in `_kwd` (like `docnm_kwd`) are typically `keyword` type and should be searched with `term` or `wildcard` queries, not full-text `multi_match`.

### Why RAGFlow's Built-in Search Works and Yours Doesn't

If you write a simplified search and get zero results while RAGFlow's `ESConnection.search()` returns matches, the gap usually comes from several places:

1. **Query preprocessing** — RAGFlow tokenizes and normalizes the query (especially for Chinese text) before hitting the `*_tks` and `*_sm_tks` fields. A raw `multi_match` with the original text won't match tokenized fields.

2. **Field boosting** — The internal search applies different weights to different fields, affecting recall and ranking.

3. **Bool query structure** — RAGFlow typically uses `bool` queries with `must/should/filter` clauses, including filters on `kb_id`, `doc_id`, `tenant_id`, and availability flags. Missing these filters changes the result set entirely.

4. **Hybrid retrieval** — The presence of `q_2560_vec` in the document source indicates vector search capability. RAGFlow may use BM25 + vector hybrid search with score fusion, while a keyword-only query will have much lower recall.

5. **Index aliases** — RAGFlow may query aliases rather than raw index names.

**The best debugging approach:** Print the exact query body that `ESConnection.search()` sends to `es.search()` and diff it against yours. The differences will be immediately clear.

---

## Part 2: Elasticsearch Is More Than Keyword Search

### Why Are ES Queries So Complex?

Elasticsearch is built on **inverted indexes + relevance scoring** — a full-text retrieval system, not just keyword matching. A real-world query needs to express:

- **Tokenization & language processing**: Chinese/English segmentation, synonyms, stopwords, spelling variants
- **Relevance ranking**: BM25 scoring, field-level boosting
- **Structured filtering**: `kb_id`, time ranges, permissions
- **Multi-strategy recall**: `must/should/minimum_should_match` for balancing precision and recall
- **Hybrid retrieval**: combining keyword and vector (semantic) search
- **Performance controls**: pagination, timeouts, sorting, `track_total_hits`, caching

What looks like "a complicated query" is really expressing: **what to search + where to search + how to score + what to filter + how to return results**.

### ChromaDB vs. Elasticsearch

ChromaDB feels simpler because it handles a narrower scope by default:

| Aspect | ChromaDB | Elasticsearch |
|--------|----------|---------------|
| Primary mode | Vector search (semantic similarity) | Full-text + structured + vector |
| Query flow | Query → embedding → nearest neighbor | Developer builds DSL queries |
| Complexity | Low — specialized tool | High — Swiss Army knife |
| Best for | Pure semantic recall | Keyword matching + filtering + aggregation + hybrid search |

If your only need is semantic retrieval, ChromaDB is simpler. If you need **exact keyword matching + permission filtering + time ranges + complex sorting + aggregation analytics**, Elasticsearch is the right tool.

### Google vs. Elasticsearch

Both use inverted indexes and relevance scoring, but they operate at completely different levels:

- **Google** is a search **product/platform** — it crawls the web, understands pages, analyzes links (PageRank), fights spam, and uses hundreds of ranking signals including quality, freshness, authority, and user behavior.
- **Elasticsearch** is a search **engine library** — it indexes *your* data and lets you define mappings, queries, and ranking rules.

Adding a web crawler to Elasticsearch gives you a basic search system, but you'd still be missing: massive crawl scheduling, content extraction, link analysis, anti-spam, query understanding (spell correction, entity recognition, intent classification), and internet-scale distributed infrastructure.

**Elasticsearch is a retrieval kernel. Google is an entire search ecosystem.**

---

## Part 3: From Docker to Kubernetes

### Docker vs. Kubernetes

| Aspect | Docker | Kubernetes |
|--------|--------|------------|
| Purpose | Build and run containers | Orchestrate containers across a cluster |
| Scope | Single machine | Multi-machine cluster |
| Features | Images, containers, networking, volumes | Scheduling, service discovery, auto-scaling, self-healing, rolling updates |
| Usage | `docker build` / `docker run` | `kubectl apply` / `helm install` |
| Best for | Development, local testing, small deployments | Production clusters, large-scale service management |

Think of Docker as **shipping containers** and Kubernetes as the **port logistics system**.

Modern Kubernetes doesn't even require Docker — it commonly uses **containerd** or **CRI-O** as the container runtime.

### Why Kubernetes for a 5-Server Setup?

With Docker alone on 5 servers, you typically:
- Install Docker on each machine manually
- Copy config files to each server
- Run `docker run` or `docker compose` individually
- Track which server runs which service
- Manually migrate workloads when a server goes down
- Update versions one machine at a time

With Kubernetes, you instead:
- **Declare desired state** ("I want 3 RAGFlow replicas, 1 MySQL, expose a Service") and Kubernetes decides *where* to place containers
- **Unify configuration** via ConfigMaps, Secrets, and container images
- **Use service names** instead of remembering IPs (`mysql`, `redis`, `ragflow`)
- **Get automatic recovery** — if a node dies, Kubernetes reschedules pods to healthy nodes
- **Roll out updates** with a single command, with automatic rollback on failure

**Docker mode:** "I tell each server what to do."  
**Kubernetes mode:** "I tell the cluster what I want, and it figures out the rest."

---

## Part 4: Helm — The Kubernetes Package Manager

Helm exists because writing raw Kubernetes YAML for complex applications is tedious. It provides:

- **Parameterized configuration** via `values.yaml`
- **Template reuse** across environments
- **Version management** with release history
- **Upgrade/rollback** operations
- **Dependency management** between sub-charts

A `helm install` or `helm upgrade` renders templates into YAML and submits them to the Kubernetes API Server. Without Kubernetes, Helm has nothing to deploy to.

For non-Kubernetes environments, use Docker Compose (single machine), systemd (traditional servers), or Ansible/Terraform (infrastructure automation).

---

## Part 5: Deploying RAGFlow on 5 Ubuntu Servers — Step by Step

### Cluster Topology

**Simple setup:**
- 1 control-plane node + 4 worker nodes

**High-availability setup:**
- 3 control-plane nodes + 2 worker nodes

### Prerequisites (All 5 Nodes)

- Ubuntu 22.04 or 24.04
- Static IP addresses
- Resolvable hostnames
- Time synchronization (NTP)
- Swap disabled
- Internal network connectivity between all nodes

### Step 1: Base System Setup (All Nodes)

```bash
sudo apt update
sudo apt install -y curl wget gnupg2 apt-transport-https \
  ca-certificates software-properties-common
```

Disable swap:

```bash
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
```

Configure kernel modules and network parameters:

```bash
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system
```

### Step 2: Install containerd (All Nodes)

```bash
sudo apt update
sudo apt install -y containerd
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' \
  /etc/containerd/config.toml
sudo systemctl restart containerd
sudo systemctl enable containerd
```

### Step 3: Install Kubernetes Components (All Nodes)

Using Kubernetes v1.29 as an example:

```bash
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | \
  sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] \
  https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | \
  sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
sudo systemctl enable kubelet
```

### Step 4: Initialize the Control Plane (Control-Plane Node Only)

```bash
sudo kubeadm init --pod-network-cidr=10.244.0.0/16
```

Configure `kubectl` for the current user:

```bash
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

Save the `kubeadm join` command from the output.

### Step 5: Join Worker Nodes (4 Worker Nodes)

Run the join command from Step 4 on each worker:

```bash
sudo kubeadm join <control-plane-ip>:6443 --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash>
```

### Step 6: Install CNI Network Plugin

Flannel for a simple start:

```bash
kubectl apply -f \
  https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

Verify:

```bash
kubectl get nodes
kubectl get pods -A
```

All nodes should show `Ready` status.

### Step 7: Install Helm (Control-Plane or Ops Machine)

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
  | bash
helm version
```

### Step 8: Install Ingress Controller

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

### Step 9: Set Up Persistent Storage

RAGFlow and its dependencies require persistent volumes. Options:

- **NFS** — Set up an NFS server and install an NFS provisioner in Kubernetes
- **Longhorn** — Better suited for multi-node clusters with dynamic PVC allocation

### Step 10: Deploy RAGFlow

**Default deployment** (with built-in dependencies):

```bash
helm upgrade --install ragflow ./helm \
  --namespace ragflow --create-namespace
```

**Production deployment** (with external services):

Create a `values.override.yaml`:

```yaml
mysql:
  enabled: false
minio:
  enabled: false
redis:
  enabled: false

env:
  MYSQL_HOST: mydb.example.com
  MYSQL_PORT: "3306"
  MYSQL_USER: root
  MYSQL_DBNAME: rag_flow
  MYSQL_PASSWORD: "your-pass"

  MINIO_HOST: s3.example.com
  MINIO_PORT: "9000"
  MINIO_ROOT_USER: rag_flow
  MINIO_PASSWORD: "your-minio-pass"

  REDIS_HOST: redis.example.com
  REDIS_PORT: "6379"
  REDIS_PASSWORD: "your-redis-pass"

  DOC_ENGINE: elasticsearch
  ELASTIC_PASSWORD: "your-es-pass"
```

Apply:

```bash
helm upgrade --install ragflow ./helm \
  -n ragflow -f values.override.yaml
```

Verify:

```bash
kubectl get all -n ragflow
kubectl get pvc -n ragflow
kubectl get ingress -n ragflow
```

### Optional: Additional Cluster Components

- `metrics-server` — resource usage monitoring
- `cert-manager` — automatic TLS certificate management
- Kubernetes Dashboard — web UI for cluster management
- Prometheus + Grafana — monitoring and alerting stack

---

## Quick Reference

| What to Install | Where |
|-----------------|-------|
| Ubuntu + containerd + kubelet + kubeadm | All 5 nodes |
| kubectl + helm | Control-plane / ops machine |
| CNI plugin (Flannel/Calico) | Cluster-wide |
| Ingress Controller | Cluster-wide |
| Persistent storage (NFS/Longhorn) | Cluster-wide |
| RAGFlow Helm Chart | Cluster-wide |

---

## Conclusion

The journey from "writing raw Elasticsearch queries" to "deploying RAGFlow on a Kubernetes cluster" spans the full stack of a modern RAG system. Understanding how search actually works under the hood — tokenization, field mapping, hybrid retrieval — is just as important as getting the infrastructure right with Kubernetes and Helm. Start with the search fundamentals, then scale up with confidence.

