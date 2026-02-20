# Web

## [FastAPI](fast-api.md)

## [Nginx](nginx.md)

## [Heatmap](https://github.com/guominjia/learn/tree/code_study/web)

## [W3School](https://www.w3school.com.cn/)

**Q**: [Why remove following line will make the number disappear?](https://www.w3school.com.cn/tiy/t.asp?f=graphics_canvas_clock_start)
```javascript
// drawFace
  ctx.beginPath();
  ctx.arc(0, 0, radius*0.1, 0, 2*Math.PI);
  ctx.fillStyle = '#333';
  ctx.fill();
```
**A**: There are no set `fillStyle` before `ctx.fillText` of `drawNumbers`, so the last `fillStyle` is `white` when `ctx.fillText`, so can not see the number in white background.

### [SVG](https://www.w3school.com.cn/graphics/svg_intro.asp)
```xml
<svg xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="5" orient="auto">
      <path d="M0 0 L10 5 L0 10 Z" fill="black" />
    </marker>
  </defs>
  <path d="m10,41l0,50.5l449,0l0,50.5" fill="none" stroke="var(--col-line)" marker-end="url(#arrow)" />
</svg>
```

### Fetch
```html
const response = await fetch('http://example.com', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ type })
});
```

## [Nodejs](https://nodejs.org/en)

Set proxy
```shell proxy.sh
npm config set proxy http://your-proxy-address:port
npm config set https-proxy http://your-proxy-address:port

npm config get proxy
npm config get https-proxy

# OR
export HTTP_PROXY=http://your-proxy-address:port
export HTTPS_PROXY=http://your-proxy-address:port
```

## [Vite](https://vite.dev/)

```bash
npm create vite@latest my-blog -- --template react
```

## [MUI](https://mui.com/material-ui/)

## [Tailwind](https://tailwindcss.com/)

| | **@mui/material** | **Tailwind CSS** |
|---|---|---|
| 类型 | 组件库 | CSS 工具类框架 |
| 提供内容 | 现成组件（Button, Card...） | CSS 类名工具 |
| 样式方式 | `sx` prop / theme | className |
| 学习曲线 | 中 | 低 |
| 定制难度 | 需要覆盖主题 | 非常灵活 |