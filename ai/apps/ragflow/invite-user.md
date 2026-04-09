# How RAGFlow Handles User Invitations and Multi-Tenant Access Control

RAGFlow implements a sophisticated multi-tenant architecture that goes far beyond simply adding a join table. Understanding the full permission model is essential for building team workflows on top of RAGFlow. This post breaks down the invitation flow, dataset access control, and model sharing mechanics discovered through source-level analysis.

---

## The UserTenant Table: More Than a Join Table

At the core of the permission system is the `UserTenant` model, defined in `api/db/db_models.py`:

```python
class UserTenant(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    user_id = CharField(max_length=32, null=False, index=True)
    tenant_id = CharField(max_length=32, null=False, index=True)
    role = CharField(max_length=32, null=False, help_text="UserTenantRole", index=True)
    invited_by = CharField(max_length=32, null=False, index=True)
    status = CharField(max_length=1, null=True, default="1", index=True)

    class Meta:
        db_table = "user_tenant"
```

Key fields to notice:

- **`role`**  not a binary flag but an enum: `owner`, `admin`, `normal`, and `invite`. The `invite` role is a *pending* state.
- **`invited_by`**  tracks the chain of trust back to the inviting user.
- **`status`**  soft-delete flag (`0` = wasted, `1` = valid).

> Every user who registers also automatically gets their own tenant entry. The user's `user_id` doubles as their `tenant_id`, and they are assigned the `owner` role from the start.

---

## The Invitation Flow: A Two-Step Acceptance Protocol

Inviting a user is not a single atomic operation. It is a deliberate two-step protocol:

### Step 1  The Owner Sends an Invitation

The tenant owner calls the invite API. This creates a `UserTenant` record with `role = UserTenantRole.INVITE`:

```python
UserTenantService.save(
    id=get_uuid(),
    user_id=user_id_to_invite,
    tenant_id=tenant_id,
    invited_by=current_user.id,
    role=UserTenantRole.INVITE,
    status=StatusEnum.VALID.value
)
```

At this point the invited user is *visible* to the system but has no meaningful access.

### Step 2  The Invitee Accepts

When the invited user accepts, their role is promoted from `INVITE` to `NORMAL`:

```python
UserTenantService.filter_update(
    [UserTenant.tenant_id == tenant_id, UserTenant.user_id == current_user.id],
    {"role": UserTenantRole.NORMAL}
)
```

Only after this step can the user consume team resources.

> **Important restriction:** Only the tenant owner can send invitations. Regular members (`normal` role) do not have the ability to invite others.

---

## Dataset Access Control: The Permission Field Matters

Joining a team does not automatically grant access to all of the owner's datasets. Dataset visibility is governed by a separate `permission` field on the `Knowledgebase` model.

The query that lists accessible knowledge bases filters on two conditions simultaneously:

```python
(cls.model.tenant_id.in_(joined_tenant_ids)
 & (cls.model.permission == TenantPermission.TEAM.value))
| (cls.model.tenant_id == user_id)
```

This means:
- A dataset is visible to team members **only** when its `permission` is set to `"Team"`.
- Datasets left at the default `"Me"` (private) remain invisible to everyone except the owner.

The `accessible()` helper enforces this at the row level:

```python
def accessible(cls, kb_id, user_id):
    docs = cls.model.select(
        cls.model.id
    ).join(
        UserTenant, on=(UserTenant.tenant_id == Knowledgebase.tenant_id)
    ).where(
        cls.model.id == kb_id,
        UserTenant.user_id == user_id
    ).paginate(0, 1)

    return bool(docs.dicts())
```

The JOIN against `UserTenant` ensures only users with an active membership record can pass the check.

---

## Model Access Control: Community vs. Enterprise

Model sharing follows a tiered policy:

| Feature | Community Edition | Enterprise Edition |
|---|---|---|
| Use tenant's default model config | Yes | Yes |
| Share custom/fine-tuned models with team | No | Yes |

Team members in the community edition can use the owner's selected base models but cannot access any custom models the owner may have configured. Custom model sharing is an enterprise-only capability.

---

## What Team Members Can Do After Joining

Once a user accepts an invitation and their role becomes `normal`, they can:

- Upload documents to the team owner's **shared** datasets (permission = Team).
- Trigger document parsing jobs inside those datasets.
- Use the team owner's shared Agents.

They **cannot**:
- Invite other users to the team.
- Access datasets with private permission.
- Share custom models (community edition).

---

## Key Takeaways

1. **The `UserTenant` table is a state machine**, not just a mapping. The `invite` role represents a pending state, and acceptance transitions the record to `normal`.
2. **Dataset sharing is opt-in per dataset.** Joining a team does not open up all resources automatically  owners must explicitly set each dataset's permission to `"Team"`.
3. **Access is enforced via SQL JOIN**, not application-layer checks alone, making it harder to accidentally bypass.
4. **Invitation authority is strictly centralized** at the owner level, which simplifies audit trails at the cost of flexibility.

Understanding these mechanics is critical before designing any automation or integration that involves multi-user RAGFlow deployments.

## References

- <https://deepwiki.com/search/usertenantdataset_00b520bc-34eb-4cfe-9573-291e54fa7468?mode=fast>