# Security

## [SSH](ssh.md)

## [Linux users](linux-users.md)

## [Input Safety](input-safety.md)

## **SailPoint**
- **类型**: 身份治理和管理(IGA)平台
- **用途**: 企业级身份生命周期管理、访问控制、合规审计
- **场景**: 管理员工账号、权限审批、访问审查、合规报告
- **层级**: 企业管理层面的完整解决方案

## **Kerberos**
- **类型**: 网络认证协议
- **用途**: 单点登录(SSO)，基于票据的认证
- **特点**: 使用对称密钥加密，需要可信的第三方(KDC)
- **场景**: 企业内网环境(如Windows Active Directory)
- **优势**: 无需在网络中传输密码

## **OAuth 2.0**
- **类型**: 授权框架
- **用途**: 第三方应用授权访问用户资源
- **核心**: 授权而非认证
- **场景**: "使用Google账号登录"、API访问授权
- **流程**: 颁发访问令牌(access token)给第三方应用

## **OpenID Connect (OIDC)**
- **类型**: 身份认证层(基于OAuth 2.0)
- **用途**: 用户身份验证和单点登录
- **核心**: 认证 + 授权
- **场景**: 现代Web/移动应用的身份验证
- **扩展**: 在OAuth 2.0基础上添加ID令牌(ID token)

## **Microsoft 域服务**

### **Active Directory Domain Services (AD DS)**
- **类型**: 目录服务 / 域控制器
- **用途**: 集中管理企业网络中的用户、计算机、组和策略
- **核心组件**:
  - **域控制器 (Domain Controller, DC)**: 负责认证和授权请求的服务器
  - **LDAP**: 目录信息的查询协议
  - **Kerberos**: AD 默认使用的认证协议
  - **DNS**: AD 依赖 DNS 进行服务发现
  - **Group Policy (GPO)**: 集中配置和管理客户端策略
- **场景**: 企业内网，Windows 机器加域、统一账号登录

### **Azure Active Directory / Microsoft Entra ID**
- **类型**: 云端身份与访问管理服务(IDaaS)
- **用途**: 云应用的身份认证、SSO、MFA、条件访问
- **与 AD DS 的区别**:
  - AD DS 面向内网/本地，基于 Kerberos/LDAP
  - Entra ID 面向云端/互联网，基于 OAuth 2.0/OIDC/SAML
- **混合场景**: 通过 **Azure AD Connect** 将本地 AD DS 同步到云端 Entra ID
- **场景**: Microsoft 365、Azure 资源访问、第三方 SaaS 应用 SSO

### **AD DS vs Entra ID 对比**

| 特性 | AD DS (本地) | Microsoft Entra ID (云端) |
|------|-------------|--------------------------|
| 部署位置 | 本地服务器 | 云端 (Azure) |
| 认证协议 | Kerberos / LDAP | OAuth 2.0 / OIDC / SAML |
| 管理对象 | 域内机器、用户、GPO | 云用户、应用、设备 |
| 访问场景 | 企业内网资源 | 云应用、SaaS、API |
| 加入方式 | 域加入 (Domain Join) | Azure AD Join / 混合加入 |
| 典型用途 | 传统企业 IT 基础设施 | 现代云优先组织 |

## **关键区别**

| 技术 | 主要功能 | 使用场景 |
|------|---------|---------|
| AD DS | 本地目录服务/域管理 | 企业内网统一账号与策略 |
| Entra ID | 云端身份与访问管理 | 云应用 SSO、MFA、条件访问 |
| SailPoint | 身份治理 | 企业身份生命周期管理平台 |
| Kerberos | 内网认证协议 | 企业内部 SSO (AD 内置) |
| OAuth 2.0 | 授权框架 | 第三方应用授权 |
| OpenID Connect | 认证+授权 | 现代 Web/移动应用身份验证 |

**简单类比**:
- AD DS 是公司大楼的门禁系统（内网）
- Entra ID 是云端的数字护照系统
- Kerberos 是大楼内部的通行票据机制
- SailPoint 是管理谁有哪些权限的 HR 系统
- OAuth 2.0 是授权书（允许第三方代为操作）
- OpenID Connect 是附带身份信息的授权书

## References
- [Hacker News](https://thehackernews.com/)
- [SailPoint](https://www.sailpoint.com/)
- [OAuth 2.0](https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols)
- [Active Directory Domain Services](https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/get-started/virtual-dc/active-directory-domain-services-overview)
- [Microsoft Entra ID](https://learn.microsoft.com/en-us/entra/identity/)