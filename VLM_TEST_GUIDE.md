# VLM Models Testing Demo - 使用说明

这是一个交互式的 VLM（Vision-Language Models）测试工具，可以测试以下模型：

## 支持的模型

1. **Grounding DINO** - 开放词汇目标检测
   - 输入：图片 + 文本描述（物体类别）
   - 输出：检测框 + 置信度

2. **YOLOv7** - COCO 数据集目标检测
   - 输入：图片
   - 输出：COCO 类别的检测框

3. **Mobile SAM** - 图像分割
   - 输入：图片 + 边界框
   - 输出：分割掩码

4. **BLIP2** - 视觉问答 (VQA)
   - 输入：图片 + 问题
   - 输出：答案文本

5. **BLIP2-ITM** - 图文匹配
   - 输入：图片 + 文本描述
   - 输出：匹配分数（0-1）

## 配置服务器地址

在 `behavior.yaml` 中配置云服务器的地址和端口：

```yaml
vlm_servers:
  grounding_dino:
    host: "your-cloud-server-ip"  # 修改为云服务器 IP
    port: 12181
  yolov7:
    host: "your-cloud-server-ip"
    port: 12184
  mobile_sam:
    host: "your-cloud-server-ip"
    port: 12183
  blip2:
    host: "your-cloud-server-ip"
    port: 12185
  blip2itm:
    host: "your-cloud-server-ip"
    port: 12182
```

## 使用方法

### 1. 交互式模式

```bash
conda activate vlfm_light
python test_vlm_models.py
```

按提示选择：
- 输入模型编号（1-5）
- 输入图片路径
- 根据模型输入相应的参数（文本/问题/边界框）

### 2. 快速测试模式

测试单张图片的所有模型：

```bash
python test_vlm_models.py --image /path/to/your/image.jpg
```

### 3. 自定义配置文件

```bash
python test_vlm_models.py --config my_config.yaml
```

## 示例用法

### Grounding DINO 示例
```
选择: 1
图片: test_image.jpg
文本: chair . bottle . person . laptop
```

### YOLOv7 示例
```
选择: 2
图片: test_image.jpg
```

### Mobile SAM 示例
```
选择: 3
图片: test_image.jpg
边界框: 100,100,300,300  (x1,y1,x2,y2)
```

### BLIP2 VQA 示例
```
选择: 4
图片: test_image.jpg
问题: What color is the chair?
```

### BLIP2-ITM 示例
```
选择: 5
图片: test_image.jpg
文本: a photo of a red chair in a room
```

### 一键测试所有模型
```
选择: 6
图片: test_image.jpg
```

## 输出结果

所有结果保存在 `vlm_test_results/` 目录下：

- `grounding_dino_result.jpg` - Grounding DINO 检测结果
- `yolov7_result.jpg` - YOLOv7 检测结果
- `mobile_sam_result.jpg` - 分割结果（带边界框和掩码叠加）
- `mobile_sam_mask.png` - 分割掩码（黑白图）
- `blip2_result.jpg` - VQA 结果（带问题和答案）
- `blip2itm_result.jpg` - ITM 结果（带匹配分数）

## 准备测试图片

建议准备包含以下内容的测试图片：
- 常见物体（椅子、瓶子、人、电脑等）
- 清晰的目标对象
- 适中的分辨率（建议 640x480 或更高）

示例图片可以放在项目根目录，命名为 `test_image.jpg`。

## 故障排除

### 连接错误
如果出现连接错误，检查：
1. 云服务器 IP 地址是否正确
2. 端口是否开放（防火墙设置）
3. VLM 服务器是否在云端正常运行

### 测试服务器连接
```bash
# 测试端口是否可访问
curl http://your-server-ip:12181/gdino
```

### 查看详细错误
运行时会显示详细的错误信息，包括网络请求失败的原因。

## 依赖包

确保已安装：
```bash
pip install opencv-python numpy pyyaml
```

## 快速开始

1. 配置 `behavior.yaml` 中的服务器地址
2. 准备一张测试图片 `test_image.jpg`
3. 运行：
   ```bash
   python test_vlm_models.py --image test_image.jpg
   ```
4. 查看 `vlm_test_results/` 目录中的结果

## 高级用法

### 批量测试多张图片

创建一个脚本循环测试：

```python
from test_vlm_models import VLMTester
import glob

tester = VLMTester()
for img_path in glob.glob("test_images/*.jpg"):
    image = tester.load_image(img_path)
    if image:
        tester.test_grounding_dino(image, "chair . table . person")
        # ... 其他测试
```

### 自定义可视化

修改 `test_vlm_models.py` 中的可视化函数，可以自定义：
- 颜色方案
- 文本大小
- 边界框粗细
- 掩码透明度等
