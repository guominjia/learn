---
layout: post
title: "Hosts Files vs. DNS: When You Need CNAME Records on Windows"
date: 2026-08-09
categories: [network]
tags: [dns, hosts, coredns, cname, windows]
---

A hosts file is a quick and useful way to override name resolution on one computer. It is not, however, a DNS zone. That distinction matters as soon as a local development or lab environment needs DNS record types such as `CNAME`, `MX`, or `TXT`.

This post explains what hosts files can do, why `nslookup` is the wrong tool to validate one, and how to run a small local CoreDNS resolver on Windows when actual DNS behavior is required.

## What a Hosts File Does

On Linux, the conventional hosts file is `/etc/hosts`. On Windows, it is:

```text
C:\Windows\System32\drivers\etc\hosts
```

Its format is deliberately simple: an IP address followed by one or more host names.

```text
127.0.0.1 app1.example.test
192.0.2.25 api.example.test api
::1 app1-ipv6.example.test
```

This is a static, local mapping from a name to an address. It is useful for testing a site before public DNS changes, reaching a lab service by name, or temporarily overriding a DNS result on one machine.

The format has no field for a DNS record type, TTL, delegation, or a canonical target. In other words, it cannot express this DNS record:

```text
app1.example.test. 60 IN CNAME web.example.test.
```

It also cannot create `MX`, `TXT`, `SRV`, or arbitrary DNS records. Nor does the Windows hosts file support a wildcard such as `*.example.test`: each local subdomain must be listed explicitly. A hosts file is an operating-system lookup input, not a complete DNS zone file.

## Do Not Test Hosts Entries with `nslookup`

`nslookup` sends DNS queries to a DNS server. Microsoft documents that it uses the configured DNS server when no server is supplied. Therefore, it is excellent for checking the records served by a DNS server, but it does not prove that an entry in the local hosts file is working.

For a hosts-file test, use an application that goes through the operating system resolver, for example:

```powershell
ping app1.example.test
```

For a DNS-server test, query the intended server explicitly:

```powershell
nslookup app1.example.test 127.0.0.1
nslookup -type=CNAME app1.example.test 127.0.0.1
Resolve-DnsName app1.example.test -Server 127.0.0.1
```

The difference is useful in troubleshooting: a browser or `ping` can succeed because of the hosts file while `nslookup` correctly reports that the DNS server has no such record.

## When a DNS Server Is Required

Use a manageable DNS service when clients must receive real DNS resource records or when the configuration must apply to more than one client. Common choices include:

| Option | Good fit |
|---|---|
| Windows DNS Server | A Windows Server environment with managed DNS zones |
| Router DNS service | Simple records that should be available to a home or lab network |
| DNS provider console | Public records for a domain you control |
| Technitium DNS Server | A feature-complete, open-source DNS server with a web console |
| CoreDNS | A compact, plugin-based DNS server suitable for declarative local rules |
| dnsmasq | A lightweight DNS and DHCP service, commonly run from Linux, WSL, or a container on Windows |
| AdGuard Home | A network-wide DNS service with a management interface and DNS-level filtering features |

**Acrylic DNS Proxy** is another Windows-friendly choice, but it is freeware rather than open source. Technitium DNS Server, CoreDNS, dnsmasq, and AdGuard Home are open-source projects.

## A Local CoreDNS Resolver on Windows

CoreDNS is written in Go and built from plugins. Its official releases provide precompiled binaries, and the running binary can show the plugins compiled into it:

```powershell
.\coredns.exe -plugins | Select-String template
```

The `template` plugin must be present for the CNAME examples below. The official release binaries include the standard plugins, but checking the actual executable avoids configuration surprises.

The following `Corefile` binds only to the loopback interface, returns a CNAME for names shaped as `service.region.corp.example.test`, and forwards all other queries to the listed upstream resolvers.

```text
.:53 {
	bind 127.0.0.1

	template IN A {
		match ^([^.]+)\.([^.]+)\.([^.]+)\.corp\.example\.test\.$
		answer "{{ .Name }} 60 IN CNAME {{ index .Match 2 }}.{{ index .Match 3 }}.corp.example.test."
		fallthrough
	}

	template IN AAAA {
		match ^([^.]+)\.([^.]+)\.([^.]+)\.corp\.example\.test\.$
		answer "{{ .Name }} 60 IN CNAME {{ index .Match 2 }}.{{ index .Match 3 }}.corp.example.test."
		fallthrough
	}

	forward . 192.168.0.1 {
		policy sequential
		health_check 5s
	}

	cache 300
	reload
	errors
}
```

For example, an `A` or `AAAA` query for `app.us.web.corp.example.test.` receives a CNAME to `us.web.corp.example.test.`. The regex captures the first three labels; the answer deliberately drops the first label.

`template` answers use DNS zone-file resource-record syntax. The trailing dot on the canonical target makes it fully qualified. CoreDNS evaluates the template only for matching `A` and `AAAA` questions; `fallthrough` lets unmatched names continue to the next plugin, which is the forwarder in this example.

### A Simpler Static Mapping

If only static A and AAAA-style mappings are needed, the CoreDNS `hosts` plugin is a compact option:

```text
.:53 {
	bind 127.0.0.1

	hosts {
		127.0.0.1 app1.example.test
		fallthrough
	}

	forward . 192.168.1.1
	cache 30
	log
	errors
}
```

This is still not a way to configure CNAME records. CoreDNS documents that its `hosts` plugin supports A, AAAA, and generated PTR records; use `template` or a proper zone file when a CNAME is required.

## Caching and Forwarding Details

In `cache 30`, `30` is the maximum time CoreDNS keeps a response in its cache. It does not extend a shorter DNS TTL: an upstream record with a 10-second TTL remains cacheable for only 10 seconds. Conversely, CoreDNS may apply its configured minimum cache lifetime unless the cache configuration is adjusted.

The `forward` block sends unmatched queries to upstream resolvers. `policy sequential` tries the upstreams in the listed order, and `health_check 5s` changes the health-check interval. Choose upstreams that are reachable from the computer running CoreDNS; use your network's resolver for private names rather than a public resolver.

## Make Windows Use the Local Resolver

Before using port 53, make sure that another DNS service is not already listening on it. Run CoreDNS from an elevated PowerShell session when required by local policy, then point the intended network adapter at the loopback resolver:

```powershell
netsh interface show interface
netsh interface ip set dns name="Wi-Fi" static 127.0.0.1 primary
```

Replace `Wi-Fi` with the adapter name returned by the first command. This changes DNS for that adapter only. To restore DHCP-provided DNS later:

```powershell
netsh interface ip set dns name="Wi-Fi" dhcp
```

The adapter can keep its IP address, subnet mask, gateway, and route configuration from DHCP while its DNS list is made static. Windows cannot keep DHCP-provided DNS servers and insert `127.0.0.1` ahead of them in the same adapter list. Before changing it, record the addresses that DHCP currently supplies:

```powershell
ipconfig /all
netsh interface ip show dnsservers name="Wi-Fi"
```

You may instead add a public fallback server:

```powershell
netsh interface ip add dns name="Wi-Fi" 1.1.1.1 index=2
```

Be deliberate with a fallback. If CoreDNS is not running, Windows can send queries directly to the fallback, which bypasses local rewrite rules and can yield inconsistent results.

After changing DNS records or adapter settings, clear and inspect the Windows DNS client cache as needed:

```powershell
ipconfig /flushdns
ipconfig /displaydns
```

## Run It Persistently

CoreDNS 1.14.3 and later can run as a native Windows service. The `-windows-service` flag does not install the service; it lets CoreDNS connect to the Windows Service Control Manager (SCM) after SCM starts the process. Register it from an elevated PowerShell session with `sc.exe`:

```powershell
$dir = 'C:\Tools\CoreDNS'

sc.exe create CoreDNS `
	binPath= "`\"$dir\coredns.exe`\" -conf `\"$dir\Corefile`\" -windows-service" `
	start= auto `
	DisplayName= "CoreDNS"

sc.exe start CoreDNS
sc.exe query CoreDNS
```

Use the directory that contains `coredns.exe` and `Corefile`; the service command uses absolute paths because Windows services do not start in the interactive user's working directory. The service name must be `CoreDNS`: the native implementation registers with SCM using that exact name. Running this command directly in a terminal is not a test of the native mode:

```powershell
.\coredns.exe -conf Corefile -windows-service
```

It reports that the process is not running as a Windows service, then continues as a foreground CoreDNS process. That result is expected because SCM did not launch it.

Manage the native service with standard Windows commands:

```powershell
sc.exe stop CoreDNS
sc.exe start CoreDNS
sc.exe qc CoreDNS
sc.exe delete CoreDNS
```

Only one CoreDNS process can listen on `127.0.0.1:53`. Do not create the native `CoreDNS` service while a WinSW-wrapped CoreDNS instance is running on that port. Stop and uninstall the wrapper first when replacing it with the native service.

For an older binary, or when wrapper-managed failure policies are specifically needed, WinSW remains an alternative.

Place `coredns.exe`, `Corefile`, and the renamed WinSW executable in one directory. For example, rename `WinSW-x64.exe` to `CoreDNSService.exe`, then place a same-named XML file beside it:

```xml
<service>
  <id>CoreDNS</id>
  <name>CoreDNS</name>
  <description>Local CoreDNS proxy and hostname rewrite service.</description>

  <executable>%BASE%\coredns.exe</executable>
  <arguments>-conf &quot;%BASE%\Corefile&quot;</arguments>
  <workingdirectory>%BASE%</workingdirectory>
  <startmode>Automatic</startmode>

  <onfailure action="restart" delay="10 sec" />
  <onfailure action="restart" delay="30 sec" />
  <onfailure action="none" />
</service>
```

Run the wrapper commands from an elevated PowerShell session:

```powershell
.\CoreDNSService.exe install
.\CoreDNSService.exe status
.\CoreDNSService.exe restart
```

The corresponding operational commands are `start`, `stop`, `restart`, `status`, and `uninstall`. Consult the wrapper's release and configuration documentation for version-specific behavior before deploying it on a shared or production machine.

## Verification Checklist

1. Start CoreDNS and check that it binds to `127.0.0.1:53` without a configuration error.
2. Query the local resolver directly:

   ```powershell
   nslookup -type=CNAME app.us.web.corp.example.test 127.0.0.1
   Resolve-DnsName app.us.web.corp.example.test -Type A -Server 127.0.0.1
   ```

3. Confirm that an unrelated public name is forwarded successfully.
4. Set the adapter DNS to `127.0.0.1`, flush the Windows DNS cache, and repeat the query without `-Server`.
5. Stop CoreDNS once as a failure test. A query should then fail unless a fallback DNS server has been configured.

## Takeaway

Use the hosts file for a local, static name-to-address override. Use a DNS server when the client must receive DNS semantics such as CNAME records, record TTLs, forwarding, logging, or rules shared across multiple devices. For a single Windows development computer, a loopback CoreDNS instance offers a small, scriptable way to bridge that gap.

## References

- [Microsoft: nslookup](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/nslookup) - Documents that `nslookup` queries a DNS server and supports record-type queries.
- [Microsoft: DNS in Windows and Windows Server](https://learn.microsoft.com/en-us/windows-server/networking/dns/dns-overview) - Describes Windows DNS client and server roles.
- [CoreDNS Manual](https://coredns.io/manual/toc/) - Documents Corefile server blocks, bundled plugins, binaries, and forwarding setups.
- [CoreDNS template plugin](https://coredns.io/plugins/template/) - Defines dynamic resource-record responses and shows a CNAME example.
- [CoreDNS hosts plugin](https://coredns.io/plugins/hosts/) - States the supported hosts-plugin record types and `fallthrough` behavior.
- [CoreDNS cache plugin](https://coredns.io/plugins/cache/) - Defines `cache` TTL as a maximum and documents TTL handling.
- [CoreDNS forward plugin](https://coredns.io/plugins/forward/) - Documents upstream selection policy and health-check configuration.
- [CoreDNS releases](https://github.com/coredns/coredns/releases) - Provides current release binaries and release notes, including Windows service support.
- [CoreDNS Windows service implementation](https://github.com/coredns/coredns/blob/v1.14.6/coremain/service_windows.go) - Defines the `-windows-service` flag and the required SCM service name, `CoreDNS`.
- [Technitium DNS Server](https://github.com/TechnitiumSoftware/DnsServer) - Documents its open-source, cross-platform DNS server and web console.
- [dnsmasq documentation](https://thekelleys.org.uk/dnsmasq/doc.html) - Describes dnsmasq as lightweight DNS/DHCP infrastructure under the GPL.
- [AdGuard Home](https://github.com/AdguardTeam/AdGuardHome) - Documents its open-source, network-wide DNS service and management features.
- [WinSW releases](https://github.com/winsw/winsw/releases) - Provides the Windows service wrapper releases used in the optional persistence example.
