
**`johnny-zhao.oai-compatible-copilot`**

## 设置在哪里

**`oaicopilot.models`** 设置(模型数组)里,每个模型对象的 **`context_length`** 字段就是"模型支持的最大上下文"。

- 修改方式:① 命令面板跑 `OAI Compatible: Open Configuration UI`;② 或直接编辑 `settings.json` 的 `oaicopilot.models`。
- 字段定义(`package.json`):`context_length`,`type: number`,**默认 128000**,范围 `[1000, 10000000]`。

## 为什么看到全是 128k

因为**没给每个模型单独填 `context_length`** 时就回退到默认值 `DEFAULT_CONTEXT_LENGTH = 128000`,所以所有模型都显示成一样。

计算逻辑在 `out/provideModel.js`:

```js
const contextLen = m?.context_length ?? DEFAULT_CONTEXT_LENGTH;   // 默认 128000
const maxOutput  = m?.max_completion_tokens ?? m?.max_tokens ?? DEFAULT_MAX_TOKENS; // 默认 4096
const maxInput   = Math.max(1, contextLen - maxOutput);
```

## 怎么改对

给每个模型显式填 `context_length`(以及需要的 `max_tokens`/`max_completion_tokens`)。例如:

```jsonc
"oaicopilot.models": [
  {
    "id": "glm-4.6",
    "owned_by": "zai",
    "context_length": 200000,
    "max_tokens": 16384,             // 输出上限
    "max_completion_tokens": 16384   // 新参数,优先于 max_tokens
  }
]
```

## 如果没配 `oaicopilot.models`

那模型是从 API `GET /models` 拉取的,`context_length` 取自响应里的 `providers[].context_length`(取不到同样回退 128000)。这种情况下"值不对"说明该 API 返回的 `context_length` 就不准——最简单的办法就是**在 `oaicopilot.models` 里手动把每个模型列出来、填对 `context_length`**,手动配置会优先于 API 拉取。

## 没有后台数据收集/遥测

**1. 所有网络请求只发往配置的端点**
所有 `fetch()` 的 URL 都是 `${baseUrl}/...` 形式,而 `baseUrl` 只来自:
- 全局 `oaicopilot.baseUrl`(默认 `https://router.huggingface.co/v1`)
- 或每个模型的 `baseUrl`

没有任何硬编码的第三方域名。

**2. 日志是本地文件,且默认关闭**
`out/logger.js` 把日志写到本地 `~/.copilot/oaicopilot/logs/oaicopilot-YYYYMMDD.log`,默认级别 `off`(不写)。即使开启,也只记录请求元数据(模型 id、消息数、URL、token 用量),并且**自动脱敏**——`Authorization` / `x-api-key` / `x-goog-api-key` 头只保留前 4 个字符 + `***`。日志保留 7 天自动清理。

**3. "track/collect/report" 都是本地功能**
- `reportUsage` / `usage.capture`:把 token 用量通过 `LanguageModelDataPart` 传给 VS Code,用于**状态栏和 Context Window 的 token 显示**,不是上报。
- `collectToolResultText`:把工具调用结果文本拼回消息,是本地处理。
- `versionManager.js` 的 "User-Agent to help quantify API usage":只是给请求加个 UA 头,方便 API 服务端统计,不收集你的数据。

可以开日志看它到底发了什么:
```jsonc
"oaicopilot.logLevel": "debug"
```
然后看 `~/.copilot/oaicopilot/logs/` 下的日志(注意里面会记录请求 URL 和脱敏后的头)。

关键代码在 `provider.js`:

```js
const provider = um?.owned_by;
const useGenericKey = !um?.baseUrl;   // ← 关键
const modelApiKey = await this.ensureApiKey(useGenericKey, provider);
```

`ensureApiKey(useGenericKey, provider)`:
```js
async ensureApiKey(useGenericKey, provider) {
    let apiKey;
    if (provider && provider.trim() !== "") {
        const normalizedProvider = provider.trim().toLowerCase();
        const providerKey = `oaicopilot.apiKey.${normalizedProvider}`;
        apiKey = await this.secrets.get(providerKey);   // ① 先找 provider 专属 key
        if (!apiKey && !useGenericKey) {                 // ② 没有且 useGenericKey=false 才弹窗
            ...弹窗输入并存储...
        }
    }
    if (!apiKey) {
        apiKey = await this.secrets.get("oaicopilot.apiKey");  // ③ 回退到 generic key
    }
    if (!apiKey && useGenericKey) { ...弹窗... }
    return apiKey;
}
```

## BaseUrl 与 ApiKey 优先级

### 1. BaseUrl 的优先级
```js
const baseUrl = um?.baseUrl || config.get("oaicopilot.baseUrl", "");
```
- 如果模型 `um` 匹配到了,且它**有** `baseUrl`,用模型的。
- 否则用全局 `oaicopilot.baseUrl`。

**所以如果"每个 Provider 的 BaseUrl"是配在别处(比如全局 `oaicopilot.baseUrl` 只留了一个,或者模型没匹配上),就会落到全局。**

### 2. ApiKey 的优先级
```js
const useGenericKey = !um?.baseUrl;
```
**`useGenericKey` 只取决于模型有没有 `baseUrl`,跟 ApiKey 无关!**

然后 `ensureApiKey`:
- ① 先按 `provider`(即 `owned_by`)找 `oaicopilot.apiKey.<provider>` 专属 key
- ③ 找不到就**无条件回退到全局 `oaicopilot.apiKey`**

**也就是说:即使给每个 Provider 配了专属 ApiKey,只要 `owned_by` 没对上,或者没存成 `oaicopilot.apiKey.<provider>` 这个 key 名,它就会静默用全局 key。**

### 模型匹配逻辑

模型匹配逻辑(`provider.js` 115-125 行):
```js
let um = userModels.find((um) =>
    um.id === parsedModelId.baseId &&
    ((parsedModelId.configId && um.configId === parsedModelId.configId) ||
     (!parsedModelId.configId && !um.configId)));
if (!um) { /* 再宽松匹配 baseId */ }
```

## 怎么让它用上每个 Provider 的配置

要让每个 Provider 的 BaseUrl + ApiKey 生效,必须满足:

1. **模型必须出现在 `oaicopilot.models` 里**,且 `id` 与 Copilot 里选中的模型 id 完全一致(含 `::configId` 时也要对上)。
2. **`baseUrl` 填在模型对象里**(不是只填全局)。
3. **ApiKey 存成 `oaicopilot.apiKey.<owned_by小写>`**(通过命令 `OAI Compatible: Set OAI Compatible Multi-Provider Apikey` 设置,或手动在 SecretStorage 里存),并且 `owned_by` 要填对。

例如:
```jsonc
"oaicopilot.models": [
  {
    "id": "glm-4.6",
    "owned_by": "zai",                    // ← 决定 ApiKey 用 oaicopilot.apiKey.zai
    "baseUrl": "https://api.z.ai/api/paas/v1",  // ← 决定 BaseUrl
    "context_length": 200000
  },
  {
    "id": "deepseek-chat",
    "owned_by": "deepseek",
    "baseUrl": "https://api.deepseek.com/v1",
    "context_length": 128000
  }
]
```
然后分别用 `Set OAI Compatible Multi-Provider Apikey` 设置 `zai` 和 `deepseek` 的 key。
