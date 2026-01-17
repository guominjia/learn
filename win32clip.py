import win32clipboard
from PIL import Image
import io
import pathlib

def get_clipboard_content():
    win32clipboard.OpenClipboard()
    
    try:
        # ============================
        # (1) 获取所有可用格式的列表
        # ============================
        formats = []
        current_format = 0
        while True:
            current_format = win32clipboard.EnumClipboardFormats(current_format)
            if current_format == 0:
                break
            formats.append(current_format)
            #data = win32clipboard.GetClipboardData(current_format)
            #print("Data:", data)
            #input("Pause")
        print("检测到剪切板格式:", formats)

        # ============================
        # (2) 按格式优先级提取内容
        # ============================
        content = {}
        
        # 优先检查文本内容
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            content['text'] = text
            print("文本:", text)

        # PNG图像（优先级高于普通位图）
        if False:#win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_PNG):
            png_data = win32clipboard.GetClipboardData(win32clipboard.CF_PNG)
            image = Image.open(io.BytesIO(png_data))
            content['image_png'] = image
            image.save("clipboard_image.png")
            print("已保存 PNG 图像到 clipboard_image.png")

        # 位图（截图时的格式）
        elif win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
            dib_data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
            image = Image.frombytes("RGB", (32, 32), dib_data)  # 大小需根据实际调整
            content['image_bitmap'] = image
            image.save("clipboard_bitmap.bmp")
            print("已保存位图到 clipboard_bitmap.bmp")

        # 文件列表（从资源管理器复制的文件）
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_HDROP):
            files = win32clipboard.GetClipboardData(win32clipboard.CF_HDROP)
            file_paths = [pathlib.Path(f) for f in files]
            content['files'] = file_paths
            print("文件列表:", [str(p) for p in file_paths])

        # HTML 格式（例如从网页复制的富文本）
        html_format = win32clipboard.RegisterClipboardFormat("HTML Format")
        if win32clipboard.IsClipboardFormatAvailable(html_format):
            html_data = win32clipboard.GetClipboardData(html_format)
            content['html'] = html_data.decode('utf-8')
            print("HTML 内容:", html_data[:1000])  # 输出前100字符
            with open("t.html", "wb") as f:
                f.write(html_data)
        
        return content

    finally:
        win32clipboard.CloseClipboard()

if __name__ == "__main__":
    result = get_clipboard_content()
    #print("完整结果:", result)