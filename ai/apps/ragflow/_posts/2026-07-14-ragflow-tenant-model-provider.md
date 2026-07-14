---
title: How RAGFlow v0.26.4 Replaces TenantLLM with a Provider-Instance-Model Design
categories: [ai, rag, ragflow]
tags: [ragflow, llm, database, migration, model-provider]
---

RAGFlow v0.26.4 changes how tenant-specific AI models are represented. The old `TenantLLM` model stored the provider, credentials, endpoint, model name, model type, limits, usage, and status in one row. The new primary design separates these responsibilities into three models:

- `TenantModelProvider`
- `TenantModelInstance`
- `TenantModel`

The legacy `TenantLLM` model and its `tenant_llm` table have not disappeared. They remain in the codebase to support migration and older execution paths, but new model-management code is built around the provider-instance-model hierarchy.

This post explains the new schema, why the split matters, how a model reference is resolved at runtime, and how v0.26.4 migrates existing `tenant_llm` data without immediately breaking backward compatibility.

## The Old TenantLLM Design

Before the new schema, one `tenant_llm` row combined several different concepts:

```text
tenant
	+ provider/factory
	+ model name and type
	+ API key
	+ API base URL
	+ maximum token count
	+ usage counter
	+ enabled/disabled status
```

The corresponding `TenantLLM` fields include:

| Field | Responsibility |
|---|---|
| `tenant_id` | Identifies the tenant |
| `llm_factory` | Identifies the provider |
| `llm_name` | Identifies the model |
| `model_type` | Describes chat, embedding, rerank, ASR, and other capabilities |
| `api_key` | Stores provider credentials or a JSON credential payload |
| `api_base` | Stores a custom provider endpoint |
| `max_tokens` | Stores the context limit |
| `used_tokens` | Tracks usage |
| `status` | Enables or disables the row |

This representation is easy to query, but it duplicates provider credentials when several models use the same endpoint and API key. It also makes multiple accounts or endpoints for one provider awkward because the provider, connection, and model are not independent entities.

For example, two OpenAI-compatible gateways serving the same model cannot be described cleanly by only `tenant_id`, `llm_factory`, and `llm_name`. The missing concept is a named provider **instance**.

## The New Three-Level Model

RAGFlow v0.26.4 separates ownership, connection configuration, and model capability:

```text
Tenant
	-> TenantModelProvider
			 -> TenantModelInstance
						-> TenantModel
```

The logical relationships are:

```text
tenant_model_provider.tenant_id
tenant_model_instance.provider_id -> tenant_model_provider.id
tenant_model.provider_id          -> tenant_model_provider.id
tenant_model.instance_id          -> tenant_model_instance.id
```

The fields are IDs rather than Peewee `ForeignKeyField` declarations, so referential integrity is mainly enforced by application services. Nevertheless, the IDs define a clear hierarchy.

### TenantModelProvider: Tenant Ownership

`TenantModelProvider` records that a tenant has configured a provider:

```python
class TenantModelProvider(DataBaseModel):
		id = CharField(max_length=32, primary_key=True)
		provider_name = CharField(max_length=128, null=False)
		tenant_id = CharField(max_length=32, null=False, index=True)
```

The table is `tenant_model_provider`, with a unique index on `(tenant_id, provider_name)`. A tenant therefore has at most one provider record for a given provider name.

This row answers:

> Which provider is available to this tenant?

It intentionally does not contain credentials or individual model names.

### TenantModelInstance: Credentials and Endpoints

`TenantModelInstance` represents one configured connection to a provider:

```python
class TenantModelInstance(DataBaseModel):
		id = CharField(max_length=32, primary_key=True)
		instance_name = CharField(max_length=128, null=False)
		provider_id = CharField(max_length=32, null=False)
		api_key = CharField(max_length=512, null=False)
		status = CharField(max_length=32, default="active")
		extra = CharField(max_length=512, default="{}")
```

The table is `tenant_model_instance`. Its `extra` JSON string can carry connection-level properties such as `base_url` and `region`.

This row answers:

> Which account, endpoint, or deployment of this provider should RAGFlow use?

One provider can have multiple named instances. For example:

```text
OpenAI
	-> production
	-> development

OpenAI-API-Compatible
	-> gateway-us
	-> gateway-eu
```

Credentials are stored once per connection rather than repeated for every model available through that connection.

### TenantModel: Model Identity and Capability

`TenantModel` represents a model exposed through a particular instance:

```python
class TenantModel(DataBaseModel):
		id = CharField(max_length=32, primary_key=True)
		model_name = CharField(max_length=128, null=True)
		provider_id = CharField(max_length=32, null=False)
		instance_id = CharField(max_length=32, null=False, index=True)
		model_type = CharField(max_length=32, null=False)
		status = CharField(max_length=32, default="active")
		extra = CharField(max_length=1024, default="{}")
```

The table is `tenant_model`. Model-level metadata such as `max_tokens` and `is_tools` can be stored in `extra`.

This row answers:

> Which model and capability are available through this provider instance?

The split makes status meaningful at two levels:

- Disabling an **instance** disables the connection.
- Disabling a **model** disables only that model on the connection.
- Marking a model `unsupported` prevents it from being used for an incompatible model type.

## How the Old Row Is Decomposed

The old and new fields map conceptually as follows:

| Legacy `TenantLLM` data | New location |
|---|---|
| `tenant_id` + `llm_factory` | `TenantModelProvider` |
| `api_key` + `api_base` | `TenantModelInstance` and its `extra` data |
| `llm_name` + `model_type` | `TenantModel` |
| Connection status | `TenantModelInstance.status` |
| Model status | `TenantModel.status` |
| `is_tools` | `TenantModel.extra` |
| `max_tokens` | `TenantModel.extra` or the provider model catalog |

The most important change is not a table rename. One denormalized row is decomposed into reusable entities with different lifecycles.

## Model References Now Include the Instance

The new hierarchy needs a model reference that can distinguish provider instances. RAGFlow uses this canonical form:

```text
model@instance@provider
```

For example:

```text
gpt-4o@production@OpenAI
text-embedding-3-small@default@OpenAI
qwen-plus@gateway-cn@OpenAI-API-Compatible
```

The older two-part format is still recognized:

```text
model@provider
```

For compatibility, v0.26.4 interprets it as:

```text
model@default@provider
```

Parsing is right-anchored with `rsplit("@", 2)`. This matters because a model name itself can contain `@`. The last part is treated as the provider, the second-to-last part as the instance, and everything remaining on the left as the model name.

If an old reference requests the `default` instance but no instance with that exact name exists, RAGFlow can fall back when the provider has exactly one active instance. It logs this as a legacy default-instance fallback. The fallback is deliberately narrow: choosing automatically among multiple active instances would be ambiguous.

## Runtime Resolution

The joint service in `api/db/joint_services/tenant_model_service.py` connects the three models. Its resolution path is approximately:

```text
model reference
	-> split model, instance, and provider names
	-> find TenantModelProvider by tenant and provider name
	-> find TenantModelInstance by provider and instance name
	-> find TenantModel by provider, instance, model name, and model type
	-> combine credentials and connection metadata from the instance
	-> combine capability metadata from the model
	-> build the model_config consumed by LLMBundle
```

The generated configuration retains the keys expected by the existing model drivers:

```python
{
		"llm_factory": provider.provider_name,
		"api_key": decoded_api_key,
		"llm_name": model.model_name,
		"api_base": instance_extra.get("base_url", ""),
		"model_type": requested_model_type,
		"is_tools": model_extra.get("is_tools", is_tools_from_credentials),
		"max_tokens": resolved_max_tokens,
}
```

This is an important compatibility boundary. Database lookup has changed, but `LLMBundle` and the model-driver registry can continue consuming a familiar `model_config` dictionary.

The resolver also accepts either a `tenant_model.id` or a composite model name. `resolve_model_config()` first attempts ID-based lookup and then tries provider-instance-name resolution. Default-model lookup similarly prefers the stored `tenant_model` ID when one is available and falls back to the model-name reference when necessary.

## Migration from tenant_llm

RAGFlow v0.26.4 includes `tools/scripts/mysql_migration.py` to transform existing MySQL data. The relevant stages must run in dependency order:

```text
tenant_model_provider
	-> tenant_model_instance
			 -> tenant_model
						-> model_id_config
```

### Stage 1: Create Providers

`TenantModelProviderStage` selects distinct `(tenant_id, llm_factory)` pairs from `tenant_llm` and creates one provider row for each pair.

The unique `(tenant_id, provider_name)` index makes the stage idempotent: an already migrated provider is not inserted again.

### Stage 2: Create Instances

`TenantModelInstanceStage` groups legacy records by tenant, factory, and API key, resolves the corresponding provider ID, and creates an instance named `default`.

Legacy API keys may contain JSON with an `is_tools` flag. Because tool support belongs to a model rather than a connection, the migration canonicalizes credentials by removing `is_tools` for comparison. Rows that differ only by this flag can therefore share one instance.

Legacy status values are normalized:

```text
1 / active / enable -> active
other values        -> inactive
```

### Stage 3: Create Models

`TenantModelStage` resolves each selected legacy row to the instance with the matching canonical API key, then inserts a `tenant_model` row containing the model name, provider ID, instance ID, model type, and normalized status.

If the legacy credential JSON contains `"is_tools": true`, that flag is moved to `tenant_model.extra` rather than retained as connection identity.

The migration intentionally treats provider instances and models differently. A credential creates the reusable connection; a model row records model-specific state that cannot be inferred from the connection alone.

### Stage 4: Normalize Stored References

`ModelIdConfigStage` upgrades stored two-part references:

```text
model@provider
```

to the explicit three-part form:

```text
model@default@provider
```

It scans both ordinary columns and nested JSON configuration in tables such as `tenant`, `knowledgebase`, `dialog`, `memory`, `search`, and canvas-related tables.

The migration utility runs in dry-run mode unless `--execute` is supplied. Production upgrades should back up the database, run a dry run, execute all stages in order, inspect the migration summary, and verify model access before removing any legacy assumptions from custom integrations.

## Why TenantLLM Still Exists

The presence of `TenantLLM` in v0.26.4 does not mean the architecture was unchanged. It reflects a staged compatibility strategy.

The old table remains useful for:

- Reading configurations created by older releases.
- Supplying credentials to execution paths not yet migrated to the new resolver.
- Providing source data for the migration utility.
- Preserving old model references while stored configuration is normalized.
- Supporting rolling upgrades in which not every component changes at exactly the same time.

Some code still carries names such as `TenantLLMService`, `LLM4Tenant`, and `LLMBundle`. A class name alone does not identify the active persistence model. For example, `LLM4Tenant` still creates the concrete driver from the normalized `model_config`, while newer callers obtain that configuration through the provider-instance-model joint service.

The Go implementation shows the same transition. New paths can resolve a `tenant_model` ID through `TenantModel`, `TenantModelProvider`, and `TenantModelInstance`, while explicitly labeled legacy helpers continue looking up `tenant_llm` where compatibility is required.

The correct operational interpretation is therefore:

> `TenantModelProvider`, `TenantModelInstance`, and `TenantModel` are the new model-management source of truth; `TenantLLM` is retained as a compatibility layer during the transition.

It should not be interpreted as permission for new integrations to keep writing only to `tenant_llm`. Doing so bypasses instance naming, new model-management APIs, and new ID-based resolution.

## Why the New Design Is Better

### Multiple Connections per Provider

A tenant can configure several accounts, regions, gateways, or deployments for the same provider and give each one a stable instance name.

### Less Credential Duplication

Many models can share one `TenantModelInstance`, so an API key and base URL do not need to be copied into every model row.

### Clearer Configuration Ownership

The new tables separate three questions:

| Question | Model |
|---|---|
| Does the tenant use this provider? | `TenantModelProvider` |
| Which credentials and endpoint should be used? | `TenantModelInstance` |
| Which model and capability are enabled? | `TenantModel` |

### Stable Model IDs

`TenantModel` uses a 32-character ID. Configuration can reference that ID rather than depend only on a compound display string. Names remain useful for APIs and user interfaces, while IDs provide a stable internal identity.

### Better Extension Points

Connection-level metadata belongs in instance `extra`; model-level metadata belongs in model `extra`. This is cleaner than adding more columns to a single table whenever a provider or model requires new behavior.

## Upgrade and Integration Guidance

When adapting custom code for v0.26.4:

1. Treat `tenant_model_provider`, `tenant_model_instance`, and `tenant_model` as the primary model-management tables.
2. Use the RAGFlow provider and model services instead of inserting related rows independently.
3. Store or return the explicit `model@instance@provider` form when a name-based reference is required.
4. Prefer `tenant_model.id` for stable internal references where the API supports it.
5. Do not assume credentials live on a model row; they belong to the instance.
6. Do not copy `is_tools` into instance identity; it is model-level metadata.
7. Keep legacy `tenant_llm` data until every required runtime path and custom integration has been verified.
8. Avoid writing new configuration only to `tenant_llm`, even though compatibility readers still exist.

## Summary

RAGFlow v0.26.4 replaces the single-row `TenantLLM` design with three focused models. `TenantModelProvider` associates a provider with a tenant, `TenantModelInstance` stores a named connection and its credentials, and `TenantModel` describes a model exposed through that connection.

The new canonical model name is `model@instance@provider`, while the old `model@provider` form maps to the `default` instance for compatibility. Runtime services resolve provider, instance, and model records and then assemble the legacy-shaped `model_config` expected by existing model drivers.

The MySQL migration utility decomposes existing `tenant_llm` rows into providers, instances, and models, then normalizes stored model references. `TenantLLM` remains in v0.26.4 to protect older data and execution paths, but it is no longer the design new integrations should target.

## References

- [RAGFlow v0.26.4 database models](https://github.com/infiniflow/ragflow/blob/v0.26.4/api/db/db_models.py)
- [RAGFlow v0.26.4 tenant model joint service](https://github.com/infiniflow/ragflow/blob/v0.26.4/api/db/joint_services/tenant_model_service.py)
- [RAGFlow v0.26.4 provider API service](https://github.com/infiniflow/ragflow/blob/v0.26.4/api/apps/services/provider_api_service.py)
- [RAGFlow v0.26.4 MySQL migration utility](https://github.com/infiniflow/ragflow/blob/v0.26.4/tools/scripts/mysql_migration.py)
- [RAGFlow v0.26.4 legacy tenant LLM service](https://github.com/infiniflow/ragflow/blob/v0.26.4/api/db/services/tenant_llm_service.py)
