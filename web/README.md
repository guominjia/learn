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