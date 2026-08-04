---
layout: post
title: "Fix Microsoft Entra Sign-In Failures Caused by Required User Assignment"
date: 2026-08-04
categories: [microsoft, azure, identity]
tags: [microsoft-entra-id, enterprise-applications, service-principal, app-role-assignment, powershell]
---

An application can be registered correctly and still reject a user at sign-in because of a tenant-local Microsoft Entra configuration: **Require assignment?**

This setting belongs to the application's **enterprise application** (the service principal), not its **app registration**. When user assignment is required, Microsoft Entra ID issues a user or access token only after the user or one of their assigned groups has an app-role assignment.

This article shows how to find the relevant setting, choose an appropriate access model, and inspect or update it with Microsoft Graph PowerShell. All identifiers below are placeholders; do not publish tenant IDs, client IDs, object IDs, user principal names, secrets, or access tokens in tickets or documentation.

---

## App Registration vs. Enterprise Application

The two portal experiences describe related but different directory objects:

- **App registrations** manage the application object, such as redirect URIs, credentials, API permissions, app roles, and owners.
- **Enterprise applications** manage service principals, the tenant-local instances that control who can access an application in that tenant.

If the portal view contains **App roles** and **Owners**, it is likely the app-registration experience. That view does not control the user-assignment requirement that blocks a sign-in. Open the matching enterprise application instead.

## Find the Setting in the Portal

1. Open the [Microsoft Entra admin center](https://entra.microsoft.com).
2. Select **Entra ID** > **Enterprise applications** > **All applications**.
3. Search for the application by its display name or application (client) ID. Use an internal value; do not paste it into public documentation.
4. Select the matching enterprise application.
5. Under **Manage**, select **Properties**.
6. Review **Assignment required?**.

The choice has a direct access-control effect:

- **No**: unassigned users can sign in to supported applications. They might not see the app in My Apps until assigned.
- **Yes**: only users assigned directly or through an assigned group can sign in.

When the setting is **Yes**, assign access through **Users and groups** in the same enterprise application. Select **Add user/group**, choose the intended user or group, choose an app role when the application exposes one, and save the assignment.

## Recommended Production Model

For a production application, keep **Assignment required?** enabled and assign a dedicated Entra security group to the application. This avoids per-user administration and makes access review a group-membership task.

Group-based application assignment requires Microsoft Entra ID P1 or P2, and nested groups are not supported for this assignment path. Confirm the tenant's licensing and group design before relying on this model.

## Inspect the Service Principal with Microsoft Graph PowerShell

The corresponding service-principal property is:

```text
appRoleAssignmentRequired
```

Use a client ID only from a secure local variable or approved secret store. The following read-only example retrieves the service principal and its assignment requirement without embedding a real identifier in the script:

```powershell
Connect-MgGraph -Scopes "Application.Read.All"

$appId = "<application-client-id>"
$servicePrincipal = Get-MgServicePrincipal `
	-Filter "appId eq '$appId'" `
	-Property "id,displayName,appId,appRoleAssignmentRequired"

$servicePrincipal | Select-Object Id, DisplayName, AppId, AppRoleAssignmentRequired
```

For delegated access, the signed-in administrator needs a supported Entra directory role. Microsoft documents **Application Administrator** and **Cloud Application Administrator** for this operation. Being an owner of the app registration alone does not necessarily grant permission to manage the enterprise application in every scenario.

## Disable the Requirement Deliberately

Turning off the requirement broadens the set of users who can sign in, so treat it as an access-policy change. The following example sets the property to `$false` for one enterprise application:

```powershell
Connect-MgGraph -Scopes "Application.ReadWrite.All"

$servicePrincipalId = "<enterprise-application-object-id>"
$params = @{ appRoleAssignmentRequired = $false }

Update-MgServicePrincipal `
	-ServicePrincipalId $servicePrincipalId `
	-BodyParameter $params
```

To retain the production restriction, set the value to `$true` and create the required user or group app-role assignments instead:

```powershell
$params = @{ appRoleAssignmentRequired = $true }

Update-MgServicePrincipal `
	-ServicePrincipalId "<enterprise-application-object-id>" `
	-BodyParameter $params
```

After either change, test with an account that should be allowed and one that should be denied. This confirms both the technical setting and the intended authorization policy.

## Troubleshooting Checklist

- Verify that you opened **Enterprise applications**, not **App registrations**.
- Confirm that the service principal matches the intended application (display names are not always unique).
- When assignment is required, assign the user or a supported group under **Users and groups**.
- Confirm that the user is a direct member of the assigned group; nested membership is not evaluated for application assignment.
- Request an appropriate Entra directory role if **Properties** or **Users and groups** is unavailable.
- Keep identifiers and tokens out of logs, screenshots, support requests, and published posts.

## References

- [Apps and service principals in Microsoft Entra ID](https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals): explains the application object/service-principal relationship and identifies Enterprise applications as the service-principal management experience.
- [Manage access to apps](https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/what-is-access-management): documents required user assignment, group-based assignment limits, and the sign-in behavior of assigned and unassigned users.
- [Manage users and groups assignment to an application](https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/assign-user-or-group-access-portal): provides the Enterprise applications portal path and supported assignment workflows.
- [Update servicePrincipal - Microsoft Graph v1.0](https://learn.microsoft.com/en-us/graph/api/serviceprincipal-update?view=graph-rest-1.0): documents `appRoleAssignmentRequired`, permissions, and the service-principal update operation.
