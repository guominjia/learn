# Security

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

## **关键区别**

| 技术 | 主要功能 | 使用场景 |
|------|---------|---------|
| SailPoint | 身份治理 | 企业身份管理平台 |
| Kerberos | 内网认证 | 企业内部SSO |
| OAuth 2.0 | 授权 | 第三方应用授权 |
| OpenID Connect | 认证+授权 | 现代应用身份验证 |

**简单类比**: SailPoint是管理系统，Kerberos是内部门禁卡，OAuth 2.0是授权书，OpenID Connect是带身份信息的授权书。

## References
- [Hacker News](https://thehackernews.com/)
- [SailPoint](https://www.sailpoint.com/)
- [OAuth 2.0](https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols)