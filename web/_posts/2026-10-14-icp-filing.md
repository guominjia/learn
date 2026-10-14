---
title: "Why a Mainland China Cloud Server Needs ICP Filing Before Your Domain Works"
date: 2026-10-14
categories: [web]
tags: [icp, china, cloud, domain, compliance]
---

You buy a cloud server in a mainland China region, point a domain at it, start Nginx, and the browser shows a "not filed" (未备案) page instead of your site. DNS works and the server works. The block comes from a legal requirement, and the cloud provider is the party that enforces it.

## The rule comes from regulation

Two documents define the requirement:

| Document | What it says |
| --- | --- |
| *Measures for the Administration of Internet Information Services* (State Council Order No. 292, 2000), Art. 4 | Commercial internet information services need a license. Non-commercial services need a filing. Without one, "不得从事互联网信息服务". |
| *Administrative Measures for the Filing of Non-commercial Internet Information Services* (MII Order No. 33, 2005), Art. 5 | Covers websites reached **by domain name or only by IP address**. |
| Order No. 33, Art. 18 | **Access providers** (IDC, ISP, and other telecom operators that provide access to websites) must not provide access to unfiled non-commercial services. |
| Order No. 33, Art. 22 / Art. 24 | Penalties: the site operator is ordered to correct and fined CNY 10,000, and the site is closed if they refuse. An access provider that violates Art. 18 is also fined CNY 10,000. |

The obligation is placed on the cloud provider as well as on you. That is why the provider blocks traffic itself instead of leaving the problem to the regulator.

## How the provider enforces it

Alibaba Cloud's documentation describes the behavior directly:

- If a domain resolves to an Alibaba Cloud mainland server and has no filing through Alibaba Cloud, its monitoring system **blocks access to the domain** and prompts you to file.
- If the domain is already filed with a different provider, you still have to add Alibaba Cloud as an access provider (接入备案). Until you do, access is blocked.

So DNS resolution is not where the block happens. The block applies after the domain already points at the server. A filing is also tied to an access provider, so moving to another cloud requires an access-filing change.

Alibaba Cloud does not publish how its monitoring system detects the domain. On the wire, though, the domain is visible in plaintext in two places:

- **HTTP**: the `Host` header. HTTP/1.1 clients must send it so that one IP can serve many host names ([RFC 9110 §7.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2)).
- **HTTPS**: the TLS `server_name` (SNI) extension in the ClientHello, which carries the DNS host name before encryption starts ([RFC 6066 §3](https://www.rfc-editor.org/rfc/rfc6066.html#section-3)).

### A likely mechanism (unverified)

> The following is an inference from the protocol facts above and common reports. No provider documentation confirms it, so treat it as a hypothesis.

1. A gateway at the provider's network ingress inspects inbound traffic to mainland servers.
2. For HTTP, typically on port 80, it reads the `Host` header.
3. For HTTPS, typically on port 443, it reads the SNI in the TLS ClientHello.
4. It looks up the domain in the provider's list of filed domains (接入备案).
5. If the domain is not on the list, the gateway acts on the request:
   - **HTTP**: it can answer with a redirect or a "not filed" notice page.
   - **HTTPS**: without your certificate it cannot serve a valid page, so it most likely resets the connection or fails the handshake.

If this guess is right, it has two consequences:

- Requests that carry no domain may not match the check. Examples are raw-IP access, ports the gateway does not inspect, and non-HTTP protocols such as SSH.
- Not being blocked does not make the site compliant, as the next section shows.

You can test the guess against your own unfiled server:

```bash
curl -v -H "Host: example.com" http://<server-ip>/
curl -v --resolve example.com:443:<server-ip> https://example.com/
curl -v -H "Host: example.com" http://<server-ip>:8080/
```

## Changing the port does not remove the obligation

Alibaba Cloud states this explicitly: if you use a domain with a non-standard port (anything other than 80 or 443) to reach a mainland server, the domain still has to be filed. The port does not change the filing obligation.

Using a raw IP address does not avoid it either. Order No. 33 Art. 5 explicitly covers sites "仅能通过互联网IP地址访问". Alibaba Cloud's filing system only accepts domains, so an IP-only site has to file through the local communications administration.

Whether a technical block actually triggers on a given port is a separate question from whether filing is required. Do not treat a missing block as permission.

## What does not need a filing

- **Servers outside mainland China.** Alibaba Cloud states that resources in Hong Kong (China) or overseas regions do not need ICP filing.
- **Non-website traffic.** The scope of Order No. 33 is websites that provide information services. Logging in to your own server over SSH does not provide a website to the public.

## Summary

| Question | Answer |
| --- | --- |
| Who imposes the requirement? | Regulation (Order No. 292, Order No. 33). |
| Who enforces the block? | The access provider (your cloud vendor), which is itself liable under Art. 18 and Art. 24. |
| Does a non-80/443 port avoid it? | No. The filing obligation is independent of the port. |
| Does IP-only access avoid it? | No. Art. 5 covers IP-only sites. |
| How do you avoid filing legitimately? | Host the site outside mainland China, such as in Hong Kong. |

## References

- [互联网信息服务管理办法 (State Council Order No. 292)](https://www.gov.cn/gongbao/content/2000/content_60531.htm): Art. 4, license vs. filing requirement; Art. 19, consequences of operating without a filing.
- [非经营性互联网信息服务备案管理办法 (MII Order No. 33)](https://www.gov.cn/gongbao/content/2005/content_93018.htm): Art. 5, scope including IP-only sites; Art. 18, access-provider obligation; Art. 22 and Art. 24, penalties.
- [Alibaba Cloud: What is an ICP filing?](https://www.alibabacloud.com/help/en/icp-filing/basic-icp-service/product-overview/what-is-an-icp-filing): The monitoring system blocks unfiled domains and domains not added to Alibaba Cloud access; filing applies only to mainland servers.
- [阿里云：备案服务器检查](https://help.aliyun.com/zh/icp-filing/basic-icp-service/user-guide/icp-filing-server-access-information-check): Non-standard ports do not change the filing obligation; Hong Kong and overseas resources need no filing.
- [阿里云：限制说明](https://help.aliyun.com/zh/icp-filing/basic-icp-service/product-overview/limits): Alibaba Cloud files domains only; IP-only sites file with the local communications administration.
- [RFC 9110 §7.2: Host and :authority](https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2): The `Host` header carries the target host name in HTTP requests.
- [RFC 6066 §3: Server Name Indication](https://www.rfc-editor.org/rfc/rfc6066.html#section-3): The TLS ClientHello carries the server's DNS host name.
