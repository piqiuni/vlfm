# 📝 VLFM Changelog

**[BEHAVIOR-1K Simulator Integration] - 2025-11-27 (@pi_dev)**

> _Comprehensive adapter implementation for running VLFM policies in BEHAVIOR-1K simulator with single RGB-D camera and visual odometry-based localization._

---

## ✨ 新增 (Added)

### **[Policy/Adapter]** BEHAVIOR-1K 仿真器策略适配器
- _场景/作用：_ 为缺少GPS/罗盘传感器的单目RGB-D相机仿真环境（如BEHAVIOR-1K）提供完整的导航策略支持
- _核心文件：_ `vlfm/policy/behavior_policies.py` (548行)
- _主要组件：_
  - `SimpleVisualOdometry`: 基于ORB特征的RGB-D视觉里程计，使用RANSAC估计刚体变换
  - `BehaviorMixin`: 将单相机观测转换为ITMPolicyV2所需格式的适配器层
  - `BehaviorITMPolicyV2`: 结合BehaviorMixin和ITMPolicyV2的完整策略实现
- _技术细节：_
  - 支持归一化深度 [0,1] 和米制深度自动检测
  - 累积位姿跟踪（相对episode起点）
  - 障碍物地图实时更新与边界探测
  - 模块化设计，VO组件可无缝替换为ORB-SLAM3等SLAM系统

### **[Server/WebSocket]** VLFM WebSocket服务器（BEHAVIOR环境专用）
- _场景/作用：_ 提供生产级WebSocket服务，将BehaviorITMPolicyV2部署到BEHAVIOR-1K仿真器
- _核心文件：_ `websocket_vlfm_server.py` (475行)
- _关键功能：_
  - `VLFMBehaviorPolicy`: VLFM策略包装器，处理BEHAVIOR观测格式
  - `CAMERA_CONFIGS`: 支持3种相机配置（left/right RealSense, ZED）
  - 自动RGB-D提取（处理RGBA→RGB转换、深度单位检测）
  - 动作空间映射（VLFM 2D动作→R1Pro 23D关节空间）
- _依赖项：_
  - BLIP2-ITM服务器（端口12182）需提前启动
  - 支持msgpack序列化、异步WebSocket通信
- _使用示例：_
  ```bash
  # 1. 启动BLIP2-ITM服务器
  python -m vlfm.vlm.blip2itm --port 12182
  
  # 2. 启动VLFM WebSocket服务器
  python websocket_vlfm_server.py --target chair --camera zed --visualize
  
  # 3. 启动BEHAVIOR环境客户端
  python behavior_env_web.py --host localhost --port 8000
  ```

### **[Demo/Testing]** 简易WebSocket服务器模板
- _场景/作用：_ 提供最小化WebSocket服务器实现，作为自定义策略服务器开发的起点
- _核心文件：_ `simple_websocker_server.py` (200行)
- _特性：_
  - `SimplePolicy`: 零动作策略（机器人保持静止，用于测试连接）
  - msgpack序列化支持（包含NumPy数组）
  - HTTP健康检查端点（/healthz）
  - 完整错误处理与日志记录

### **[Demo/Interactive]** 键盘控制WebSocket服务器
- _场景/作用：_ 交互式键盘控制服务器，支持实时关节控制与状态监控
- _核心文件：_ `keyboard_policy_server.py` (450行)
- _交互方式：_
  - `1, 2`: 切换关节索引（0-22）
  - `[, ]`: 调整当前关节值（±delta）
  - `r`: 重置所有关节为零
  - `h`: 显示帮助信息
  - `q`: 退出服务器
- _实现亮点：_
  - 非阻塞键盘输入（独立线程）
  - 动作持久化（关节值保持直到手动修改）
  - 实时状态显示（关节索引、当前值、步数）
  - 终端原始模式处理（termios）
---

## ♻️ 优化 (Changed)

### **[Architecture/Modularity]** 视觉里程计接口标准化
- _优化点：_ SimpleVisualOdometry采用接口设计，支持通过继承/组合替换为其他SLAM系统
- _收益：_
  - 代码解耦：VO逻辑与策略逻辑完全分离
  - 易于升级：可直接替换为ORB-SLAM3、pyslam等高精度SLAM
  - 统一接口：`estimate_motion(rgb, depth) -> (position, heading)`
- _示例替换方案：_
  ```python
  # 用ORB-SLAM3替换SimpleVisualOdometry
  from some_slam_lib import ORBSLAM3
  self._vo = ORBSLAM3(fx=camera_fx, fy=camera_fy)  # 接口兼容
  ```

### **[Observation/Format]** 统一观测格式处理
- _优化点：_ BehaviorMixin自动处理多种深度格式（归一化 vs 米制）
- _收益：_
  - 提升鲁棒性：自动检测深度范围并转换
  - 简化集成：用户无需手动归一化深度数据
- _实现细节：_
  ```python
  if depth.max() > 1.0:
      depth_normalized = (depth - min_depth) / (max_depth - min_depth)
  ```

### **[Logging/Monitoring]** 增强日志系统
- _优化点：_ 所有服务器添加分级日志（INFO/WARNING/ERROR）与步数统计
- _收益：_
  - 提升可观测性：关键事件（连接、重置、动作生成）清晰记录
  - 便于调试：异常栈信息完整输出
  - 性能监控：定期输出VO匹配点数、观测统计

---

## 🔧 技术债务 (Technical Debt)

### **[TODO/Known Issues]** 需要验证的项目
1. **R1Pro动作空间映射** (websocket_vlfm_server.py:270-272)
   - _问题：_ 当前假设索引0-1控制基座旋转与前进速度
   - _需求：_ 需参考R1Pro官方文档确认23D动作空间的实际索引定义
   - _临时方案：_ 其余21个关节设置为零

2. **视觉里程计漂移** (behavior_policies.py:95-264)
   - _问题：_ SimpleVisualOdometry在长轨迹上会累积误差
   - _影响：_ Episode时长>5分钟时定位精度可能下降
   - _推荐方案：_ 生产环境建议替换为ORB-SLAM3或pyslam

3. **相机参数自动计算** (websocket_vlfm_server.py:98-101)
   - _问题：_ 当前默认假设90° FOV计算焦距（fx = width/2）
   - _影响：_ 实际相机FOV不同时会影响深度反投影精度
   - _解决方案：_ 用户应显式传入 `--camera-fx` 和 `--camera-fy` 参数


## ⚙️ 配置参数 (Configuration)

### **[Policy/Parameters]** BehaviorITMPolicyV2可配置项
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `camera_fx` | float | 320.0 | 相机焦距x（像素） |
| `camera_fy` | float | 320.0 | 相机焦距y（像素） |
| `min_depth` | float | 0.1 | 最小有效深度（米） |
| `max_depth` | float | 10.0 | 最大有效深度（米） |
| `text_prompt` | str | "This looks like..." | ITM模型文本提示 |
| `visualize` | bool | False | 是否生成可视化输出 |

### **[Server/Arguments]** websocket_vlfm_server命令行参数
```bash
--port 8000              # 监听端口
--target chair           # 目标对象名称
--camera zed             # 使用的相机（left/right/zed）
--camera-fx 400.0        # 焦距x（可选）
--visualize              # 启用可视化
--verbose                # 详细日志模式
```

---

## 🚀 使用流程 (Workflow)

### **[完整部署流程]**
```bash
# 步骤1: 激活环境
conda activate vlfm

# 步骤2: 启动BLIP2-ITM服务器（必须）
python -m vlfm.vlm.blip2itm --port 12182

# 步骤3: 启动VLFM WebSocket服务器
python websocket_vlfm_server.py \
    --port 8000 \
    --target chair \
    --camera zed \
    --camera-fx 360.0 \
    --camera-fy 360.0 \
    --visualize

# 步骤4: 启动BEHAVIOR环境客户端
python behavior_env_web.py --host localhost --port 8000

# 可选：使用键盘控制服务器进行测试
python keyboard_policy_server.py --port 8000 --delta 0.2
```
